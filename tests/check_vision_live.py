#!/usr/bin/env python3
"""Opt-in real API image/cache/tool/stream checks; uses only synthetic fixtures.

Run against the backend and again against a router alias. Does not log keys,
URLs, image payloads, conversation text, or system paths in its numeric receipt.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url', required=True, help='OpenAI-compatible URL ending /v1')
    ap.add_argument('--model', required=True)
    auth = ap.add_mutually_exclusive_group(required=True)
    auth.add_argument('--key-file', type=Path)
    auth.add_argument('--router-config', type=Path, help='Read an existing router apiKey in memory')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--text-only', action='store_true')
    args = ap.parse_args()
    key = args.key_file.read_text().strip() if args.key_file else json.loads(args.router_config.read_text())['apiKey']
    rows = []
    def record(name, **data):
        row = {'case': name, **data}
        rows.append(row)
        args.output.write_text(json.dumps({'status':'running','cases':rows}, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    def chat(messages, *, tools=None, stream=False, max_tokens=128, temperature=0, seed=81):
        payload = {'model':args.model, 'messages':messages, 'temperature':temperature, 'seed':seed,
            'reasoning_effort':'none', 'max_tokens':max_tokens, 'stream':stream}
        if tools:
            payload['tools'] = tools
        if stream:
            payload['stream_options'] = {'include_usage':True}
        req = urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
            data=json.dumps(payload).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            if not stream:
                return json.load(response)
            events = [json.loads(line[6:]) for line in response if line.startswith(b'data: {')]
        assert not any('error' in e for e in events)
        final = next(e for e in events if e.get('choices') and e['choices'][0].get('finish_reason'))
        usage_event = next(e for e in events if e.get('usage'))
        assert final['timings'] == usage_event['timings']
        content = ''.join(c.get('delta',{}).get('content','') for e in events for c in e.get('choices',[]))
        return {'choices':[{'message':{'role':'assistant','content':content}}], 'usage':usage_event['usage'],
            'timings':final['timings'], 'x_qwentin':final.get('x_qwentin',{})}
    def text(result):
        return (result['choices'][0]['message'].get('content') or '').strip().lower().rstrip('.')
    def metrics(name, result, **extra):
        t = result['timings'];u=result['usage']
        assert t['prompt_n'] + t['cache_n'] == u['prompt_tokens']
        assert abs(t['prompt_per_second'] - t['prompt_n'] * 1000 / t['prompt_ms']) < 1e-6
        record(name, usage=u, timings=t, x_qwentin=result.get('x_qwentin',{}), **extra)
    def image(color, *, layout=False, label=None, resolution=256):
        name = ('number-'+label) if label else (('layout-' if layout else '')+color)
        if resolution != 256:
            name = str(resolution)+'-'+name
        fixture = Path(__file__).parent/'fixtures'/(name+'.png')
        if fixture.is_file():
            return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(fixture.read_bytes()).decode()}}
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new('RGB', (512,256) if label else (resolution,resolution), color)
        d = ImageDraw.Draw(im)
        if layout:
            d.rectangle((128,0,255,255), fill='blue' if color=='red' else 'red')
        if label:
            font = ImageFont.truetype('DejaVuSans-Bold.ttf', 90)
            d.text((35,55), label, fill='black', font=font)
        out = io.BytesIO();im.save(out, format='PNG')
        return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()}}
    def messages(part, question):
        return [{'role':'user','content':[part, {'type':'text','text':question}]}]

    # Same public fixture used before/after the native changes.
    for n in (200, 1700):
        prompt = 'Synthetic queue record: workers claim jobs, record durable progress, and retry transient failures.\n' * n
        prompt += '\nWrite a detailed engineering guide to durable queues. Continue for at least 1000 words.'
        r = chat([{'role':'user','content':prompt}])
        digest = hashlib.sha256(json.dumps(r['choices'][0]['message'],sort_keys=True).encode()).hexdigest()
        metrics('text_'+str(n), r, response_sha256=digest)
    if args.text_only:
        args.output.write_text(json.dumps({'status':'pass','cases':rows},indent=2)+'\n');return

    red, blue = image('red'), image('blue')
    question = 'What is the dominant color? Reply with only red or blue.'
    for color, part, stream in [('red',red,False), ('blue',blue,True), ('red',red,True)]:
        r = chat(messages(part,question), stream=stream)
        assert text(r) == color, 'Color fixture failed'
        metrics('color_'+color+('_stream' if stream else ''), r, answer_correct=True)
    for color, side in [('red','left'), ('blue','right')]:
        r = chat(messages(image(color,layout=True),'Which side of the image is red? Reply with only left or right.'))
        assert text(r) == side, 'Spatial fixture failed'
        metrics('red_on_'+side, r, answer_correct=True)
    r = chat(messages(image('white',label='8472'),'Read the four-digit number in this image. Reply only with the digits.'))
    assert text(r) == '8472', 'OCR fixture failed'
    metrics('ocr',r,answer_correct=True)
    r = chat([{'role':'user','content':[blue,red,{'type':'text','text':
        'Give the dominant colors of the first image and second image, in that order. Reply only blue,red or red,blue.'}]}])
    assert text(r).replace(' ','') == 'blue,red', 'Multiple-image order failed'
    metrics('multiple_images',r,answer_correct=True)
    r = chat(messages(image('blue',resolution=2048),question))
    assert text(r)=='blue' and r['x_qwentin']['vision']['image_tokens']==4096
    metrics('large_image_4096_visual_tokens',r,answer_correct=True)

    tools=[{'type':'function','function':{'name':'record_color','description':'Record the observed color.',
        'parameters':{'type':'object','properties':{'color':{'type':'string','enum':['red','blue']}},'required':['color']}}}]
    ms = messages(red, 'Call record_color with the dominant color of this image. Use the tool now.')
    r = chat(ms, tools=tools)
    assistant = r['choices'][0]['message']
    call = assistant['tool_calls'][0]
    assert call['function']['name']=='record_color' and json.loads(call['function']['arguments'])['color']=='red'
    metrics('image_tool_call',r,answer_correct=True)
    ms += [assistant, {'role':'tool','tool_call_id':call['id'],'content':'The color was successfully recorded.'},
        {'role':'user','content':'Which color did you just observe? Reply with only the color.'}]
    follow = chat(ms, tools=tools, stream=True)
    assert text(follow)=='red'
    expected_cache=r['usage']['total_tokens']-1
    assert follow['timings']['cache_n']==expected_cache, 'Image/tool continuation did not reuse full consumed prefix'
    metrics('image_tool_followup',follow,answer_correct=True,full_consumed_prefix_reused=True)
    appended = ms + [follow['choices'][0]['message'], {'role':'user','content':[blue,
        {'type':'text','text':'What is the dominant color of the latest image I just attached? Reply with only the color.'}]}]
    added = chat(appended,tools=tools)
    assert text(added)=='blue'
    assert added['timings']['cache_n']==follow['usage']['total_tokens']-1, 'Appending an image lost the existing prefix'
    metrics('appended_image',added,answer_correct=True,full_consumed_prefix_reused=True)
    # Same token layout with different pixels must not match the old negative IDs.
    ms[0]['content'][0]=blue
    ms[-1]['content']='Look at the actual image attached to the first message again. Ignore the tool record. What color is the image? Reply with only the color.'
    changed = chat(ms,tools=tools)
    assert text(changed)=='blue' and changed['timings']['cache_n']==0, 'Stale image KV reused'
    metrics('replaced_image',changed,answer_correct=True,cache_invalidated=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(chat,messages(part,question)) for part in [red,blue]]
        for expected, future in zip(['red','blue'],futures):
            r=future.result();assert text(r)==expected
            metrics('queued_'+expected,r,answer_correct=True)
    r=chat([{'role':'user','content':'What is six times seven? Reply with only the integer.'}])
    assert text(r)=='42'
    metrics('text_after_images',r,answer_correct=True)
    stochastic = [chat(messages(blue, 'Describe the image in one sentence.'), temperature=1, seed=817) for _ in range(2)]
    assert stochastic[0]['choices'][0]['message']==stochastic[1]['choices'][0]['message']
    assert 'blue' in text(stochastic[0])
    metrics('seeded_stochastic_image',stochastic[1],answer_correct=True,repeat_identical=True)
    try:
        chat(messages({'type':'image_url','image_url':{'url':'http://127.0.0.1/private'}},question))
    except urllib.error.HTTPError as e:
        assert e.code==400
        record('network_image_rejected',http_status=e.code)
    else:
        raise AssertionError('Network image was fetched or accepted')
    args.output.write_text(json.dumps({'status':'pass','cases':rows},indent=2)+'\n')


if __name__=='__main__':
    main()
