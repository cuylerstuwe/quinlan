"""On-device sampler oracle, distribution, boundary and reproducibility tests."""
import ctypes as C
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

lib = C.CDLL(sys.argv[1])
lib.quinlan_sampling_configure.argtypes = [C.c_int,C.c_float,C.c_float,C.c_int,C.c_float,C.c_uint64,C.c_int]
lib.quinlan_sampling_configure.restype = C.c_int
lib.quinlan_sampling_apply.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int),C.c_void_p]
lib.quinlan_sampling_apply.restype = C.c_int
lib.quinlan_spec_test.argtypes = [C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_int),C.c_int,C.c_int,C.c_uint64,C.c_void_p]
lib.quinlan_spec_test.restype = C.c_int
lib.quinlan_spec_pipeline_test.argtypes=[C.POINTER(C.c_int),C.POINTER(C.c_int),C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p]
lib.quinlan_spec_pipeline_test.restype=C.c_int
MASK = 2**64-1
def mix(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)
def uniform(seed, pos):
    return ((mix(seed ^ mix(pos)) >> 11) + .5) / 2**53
def oracle(row, params):
    T,p,k,mp,seed = params
    order = np.argsort(-row, kind='stable')
    weights = np.exp((row[order].astype(np.float64)-float(row[order[0]])) / np.float32(T))
    if k:
        weights[k:] = 0
    cdf = np.cumsum(weights)
    end = np.searchsorted(cdf, np.float32(p)*cdf[-1], side='left')
    if mp:
        valid = np.flatnonzero(weights[:end+1] >= np.float32(mp))
        end = valid[-1] if len(valid) else 0
    prob = weights.copy(); prob[end+1:] = 0; prob /= prob.sum()
    return order, np.cumsum(prob), prob
def run(rows, params, positions=None):
    rows = np.asarray(rows, dtype=np.float32)
    N,V = rows.shape
    logits = torch.tensor(rows, device='cuda')
    # T=0 must preserve upstream argmax exactly, including stable ties.
    out = logits.argmax(-1).to(torch.int32)
    slots = (C.c_int*N)(*([0]*N))
    positions = list(range(N)) if positions is None else positions
    pos = (C.c_int*N)(*[p-1 for p in positions])
    T,p,k,mp,seed = params
    assert lib.quinlan_sampling_configure(0,T,p,k,mp,seed,0) == 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    rc = lib.quinlan_sampling_apply(out.data_ptr(), logits.data_ptr(), V, N, slots, pos, None)
    torch.cuda.synchronize()
    assert rc == 0, rc
    return out.cpu().numpy(), time.perf_counter()-start

reports=[]
base=np.array([2.5,2,1,0.5,0,-1,-4,-np.inf],dtype=np.float32)
cases=[(1,1,0,0,23),(.7,1,0,0,9),(1,.65,0,0,17),(1,1,3,0,19),
       (1,.9,6,.4,23),(2,1,0,0,2**64-1),(.01,.01,1,1,0)]
for params in cases:
    N=2048
    out,dt=run(np.broadcast_to(base,(N,len(base))).copy(),params)
    order,cdf,prob=oracle(base,params)
    expected=order[np.searchsorted(cdf,[uniform(params[-1],p) for p in range(N)],side='right')]
    assert np.array_equal(out,expected),(params,np.flatnonzero(out!=expected)[:10])
    counts=np.bincount(out,minlength=len(base))/N
    probs=np.zeros(len(base));probs[order]=prob
    assert abs(counts-probs).max()<.05,(counts,probs)
    reports.append({'parameters':params,'draws':N,'oracle_matches':True,'max_distribution_error':float(abs(counts-probs).max()),'seconds':dt})

for row in [np.zeros(8), np.array([0,-np.inf,-np.inf]), np.array([-900,-901,-910])]:
    params=(1,.75,0,.05,31)
    out,_=run(np.broadcast_to(row,(512,len(row))).copy(),params)
    order,cdf,_=oracle(row.astype(np.float32),params)
    expected=order[np.searchsorted(cdf,[uniform(31,p) for p in range(512)],side='right')]
    assert np.array_equal(out,expected)

rng=np.random.default_rng(42)
rows=rng.normal(0,3,(8,248320)).astype(np.float32)
params=(1,.95,20,.05,9876)
out,dt=run(rows,params)
again,_=run(rows,params)
assert np.array_equal(out,again)
for i,row in enumerate(rows):
    order,cdf,_=oracle(row,params)
    expected=order[np.searchsorted(cdf,uniform(params[-1],i),side='right')]
    assert out[i]==expected,(i,out[i],expected)
greedy,_=run(rows,(0,.1,1,1,0))
assert np.array_equal(greedy,rows.argmax(-1))
# Draws depend on request seed + output position, not batch partition or rejected suffix length.
for i in range(len(rows)):
    one,_=run(rows[i:i+1],params,[i])
    assert one[0]==out[i]
for vals in [(0,-1,1,0,0,0,0),(0,1,0,0,0,0,0),(0,1,1,-2,0,0,0),(0,1,1,0,float('nan'),0,0)]:
    assert lib.quinlan_sampling_configure(*vals)<0

# Deterministic proposal validation: match target draws until rejection, then emit the target
# draw at that position. This must equal sequential target sampling for every seed.
P=np.array([[.6,.3,.1],[.2,.5,.3],[.4,.4,.2]])
for seed in range(10000):
    def target(prefix):
        row=P[(sum(prefix)+len(prefix))%len(P)]
        return int(np.searchsorted(row.cumsum(),uniform(seed,len(prefix)),side='right'))
    plain=[]
    for i in range(12): plain.append(target(plain))
    spec=[]
    while len(spec)<12:
        draft=[]
        for k in range(4):
            prefix=spec+draft
            draft.append(int(P[(sum(prefix)+len(prefix))%len(P)].argmax()))
        for proposal in draft:
            sampled=target(spec)
            spec.append(sampled)
            if sampled!=proposal or len(spec)>=12:break
    assert spec==plain

# Exact device acceptance/residual oracle, including disjoint supports and p=q.
accept_reports=[]
for target,draft in [([.6,.3,.1],[.2,.5,.3]),([.1,.2,.7],[.7,.2,.1]),
                     ([.5,0,.5],[0,1,0]),([.3,.3,.4],[.3,.3,.4]),
                     ([.01,.09,.9],[.98,.01,.01])]:
    N=4096; seed=456
    p=np.array(target,dtype=np.float32);q=np.array(draft,dtype=np.float32)
    drafts=np.searchsorted(q.astype(np.float64).cumsum(),np.random.default_rng(37).random(N),side='right')
    dp=torch.tensor(np.broadcast_to(p,(N,3)).copy(),device='cuda')
    dq=torch.tensor(np.broadcast_to(q,(N,3)).copy(),device='cuda')
    out=torch.empty(N,device='cuda',dtype=torch.int32)
    ds=(C.c_int*N)(*drafts)
    torch.cuda.synchronize()
    rc=lib.quinlan_spec_test(out.data_ptr(),dp.data_ptr(),dq.data_ptr(),ds,3,N,seed,None)
    assert rc==0,rc
    got=out.cpu().numpy();expected=[];accepted=0
    residual=np.maximum(p-q,0).astype(np.float32)
    for i,d in enumerate(drafts):
        ua=uniform(seed^0x414343455054524e,i)
        if ua*float(q[d])<float(p[d]): expected.append(d);accepted+=1
        else:
            cdf=residual.cumsum(dtype=np.float32)
            ur=uniform(seed^0x524553494455524e,i)
            expected.append(int(np.searchsorted(cdf.astype(np.float64),ur*float(cdf[-1]),side='right')))
    assert np.array_equal(got,expected),np.flatnonzero(got!=expected)[:10]
    error=float(abs(np.bincount(got,minlength=3)/N-p).max())
    assert error<.05,error
    accept_reports.append({'p':target,'q':draft,'oracle_matches':True,'acceptance':accepted/N,
        'expected_acceptance':float(np.minimum(p,q).sum()),'distribution_error':error})

pipeline_cases=0
for full_V,draft_V in [(8,3),(248320,65536)]:
    K=3;seed=8391;params=(1,.95,20,.05,seed)
    mapping=np.linspace(0,full_V-1,draft_V,dtype=np.int32)
    dl=rng.normal(0,2,(K,draft_V)).astype(np.float32)
    tl=rng.normal(0,2,(K+1,full_V)).astype(np.float32)
    d_dl=torch.tensor(dl,device='cuda');d_tl=torch.tensor(tl,device='cuda')
    d_map=torch.tensor(mapping.astype(np.float32),device='cuda')
    drafts=(C.c_int*K)();verified=(C.c_int*(K+1))()
    assert lib.quinlan_sampling_configure(0,*params[:4],seed,1)==0
    torch.cuda.synchronize()
    assert lib.quinlan_spec_pipeline_test(drafts,verified,d_dl.data_ptr(),d_tl.data_ptr(),d_map.data_ptr(),draft_V,full_V,K,None)==0
    for k in range(K+1):
        po,pcdf,pweights=oracle(tl[k],params)
        p=np.zeros(full_V);p[po]=pweights
        index=k+1
        if k<K:
            qo,qcdf,qweights=oracle(dl[k],params)
            qi=int(qo[np.searchsorted(qcdf,uniform(seed^0x44524146545f524e,index),side='right')])
            assert drafts[k]==mapping[qi]
            q=np.zeros(full_V);q[mapping[qo]]=qweights
            d=int(drafts[k])
            if uniform(seed^0x414343455054524e,index)*q[d]<p[d]:expected=d
            else:
                residual=np.maximum(p-q,0);cdf=residual.cumsum()
                expected=int(np.searchsorted(cdf,uniform(seed^0x524553494455524e,index)*cdf[-1],side='right'))
        else:expected=int(po[np.searchsorted(pcdf,uniform(seed,index),side='right')])
        assert verified[k]==expected,(full_V,k,verified[k],expected)
    pipeline_cases+=1
print(json.dumps({'status':'pass','device':torch.cuda.get_device_name(),'cases':reports,
    'full_vocab_rows':8,'full_vocab_ms_per_row':dt/8*1000,'toy_spec_seeds':10000,
    'greedy_unchanged':True,'batch_partition_reproducible':True,'probability_verifier':accept_reports,
    'reduced_vocab_pipeline_cases':pipeline_cases},indent=2))
