"""Independent Transformers position oracle and device bridge checks.

Call check(engine) before exposing a newly created, idle vision-enabled engine.
Uses small synthetic tensors; never instantiates a second language model.
"""
import ctypes
from types import SimpleNamespace


def check(engine, model_dir):
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextRotaryEmbedding
    from quinlan.vision import position_ids
    config = Qwen3_5Config(**json.loads((Path(model_dir)/'config.json').read_text()))
    owner = SimpleNamespace(config=config)
    owner.get_vision_position_ids = lambda *a, **kw: Qwen3_5Model.get_vision_position_ids(owner,*a,**kw)
    image_id = config.image_token_id
    for grids in [[(1,16,16)],[(1,32,48)],[(1,16,16),(1,24,32)]]:
        ids=[21,22,23]; row=0
        for t,h,w in grids:
            n=t*h*w//4
            ids += list(range(-row-1,-row-n-1,-1))+[24,25,26]
            row+=n
        actual,delta=position_ids(ids,grids)
        hf_ids=torch.tensor([[image_id if x<0 else x for x in ids]])
        types=(hf_ids==image_id).int()
        expected,expected_delta=Qwen3_5Model.get_rope_index(owner,hf_ids,types,image_grid_thw=torch.tensor(grids))
        np.testing.assert_array_equal(actual,expected[:,0,:].T.numpy())
        assert delta==expected_delta.item()
    lib=engine.L
    probe=lib.quinlan_vision_probe
    probe.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    probe.restype=ctypes.c_int
    embedding=torch.zeros((2,5120),dtype=torch.bfloat16,device='cuda')
    embedding[:,0]=torch.tensor([1.25,-2.5],device='cuda')
    torch.cuda.synchronize()
    coords=np.array([[3,7,11],[4,8,12],[5,9,13],[6,10,14],[7,11,15],[8,12,16]],dtype=np.int32)
    assert lib.quinlan_vision_configure(embedding.data_ptr(),2,coords.ctypes.data,len(coords),-2)==0
    out=np.empty((8,32),dtype=np.int32)
    assert probe(0,8,out.ctypes.data,None)==0
    # Use the reference model's frequency recomposition, not another copy of
    # the native axis-selection code, as the oracle for all 32 frequencies.
    expanded=torch.tensor(coords.T[:,None,:,None]).expand(3,1,6,32).clone()
    rotary=Qwen3_5TextRotaryEmbedding(config.text_config)
    expected=rotary.recomposition_frequencies(expanded)[0,:,:32].numpy()
    np.testing.assert_array_equal(out[:6],expected)
    np.testing.assert_array_equal(out[6:],np.array([[4]*32,[5]*32]))
    feature_out=np.empty(2,dtype=np.float32)
    short_out=np.empty((2,32),dtype=np.int32)
    assert probe(0,2,short_out.ctypes.data,feature_out.ctypes.data)==0
    np.testing.assert_array_equal(feature_out,[1.25,-2.5])
    assert lib.quinlan_vision_configure(None,0,None,0,0)==0
    assert probe(0,8,out.ctypes.data,None)==0
    np.testing.assert_array_equal(out,np.arange(8)[:,None].repeat(32,axis=1))
    return {'position_grids':3,'device_frequencies_checked':32,'visual_embedding_rows_checked':2,
            'generated_position_delta':True,'text_reset':True,'status':'pass'}
