"""Real-tokenizer cache regressions; CPU-only, no inference or CUDA context."""
import copy
import json
import os
from pathlib import Path
import unittest

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
from quinlan import server as service
from quinlan.tool_parser import parse_tool_calls


@unittest.skipUnless(AutoTokenizer and os.environ.get("QWENTIN_MODEL_DIR"), "set QWENTIN_MODEL_DIR for real-tokenizer tests")
class TokenHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(os.environ["QWENTIN_MODEL_DIR"], local_files_only=True)
        cls.parse = staticmethod(parse_tool_calls)
        cls.eos = {cls.tok.eos_token_id, cls.tok.convert_tokens_to_ids("<|im_end|>"),
                   cls.tok.convert_tokens_to_ids("<|endoftext|>")}

    def setUp(self):
        self.cache = service.TokenHistory()
        self.opts = service.template_options({"reasoning_effort": "high"})
        self.tools = [{"type": "function", "function": {"name": "inspect", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "options": {"type": "object"}}}}}]
        self.messages = [{"role": "system", "content": "Synthetic history preservation test."},
                         {"role": "user", "content": "Synthetic cache context.\n" * 3000 + "Inspect fixture.txt."}]
        self.xml = '<tool_call>\n<function=inspect>\n<parameter=path>\nfixture.txt\n</parameter>\n</function>\n</tool_call>'

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def render(self, messages, tools=None, opts=None, generation=True):
        return self.tok.apply_chat_template(service.prepare_messages(messages), tools=self.tools if tools is None else tools,
            tokenize=False, add_generation_prompt=generation, **(self.opts if opts is None else opts))

    def response(self, raw, messages=None, terminated=True):
        messages = service.prepare_messages(self.messages if messages is None else messages)
        prompt = self.render(messages)
        ids, _ = self.cache.resolve(self.tok, messages, self.tools, self.opts, prompt)
        out = self.encode(raw + ("<|im_end|>" if terminated else ""))
        reasoning, text = service.split_reasoning(raw, True, final=True)
        content, calls = self.parse(text, self.tools)
        assistant = {"role": "assistant", "content": content or None}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        if calls:
            assistant["tool_calls"] = calls
        self.assertTrue(self.cache.remember(self.tok, messages, self.tools, self.opts, ids, out, self.eos, assistant))
        return messages + [assistant], ids + out

    def continuation(self, messages, tools=None, opts=None):
        normalized = service.prepare_messages(messages)
        return self.cache.resolve(self.tok, normalized, self.tools if tools is None else tools,
            self.opts if opts is None else opts, self.render(normalized, tools, opts))

    def add_result(self, messages):
        return messages + [{"role": "tool", "tool_call_id": messages[-1]["tool_calls"][0]["id"],
                            "content": 'Synthetic successful result: {"total":9}'}]

    def test_formatting_variants_keep_every_original_token(self):
        cases = {
            "canonical_xml": 'Inspect the file.\n</think>\n\n' + self.xml,
            "extra_reasoning_newline": 'Inspect the file.\n\n</think>\n\n' + self.xml,
            "compact_xml": 'Inspect the file.\n</think>\n\n' + self.xml.replace('<tool_call>\n', '<tool_call>').replace('<function=inspect>\n', '<function=inspect>'),
            "json_tool_call": 'Inspect the file.\n</think>\n\n<tool_call>{"name":"inspect","arguments":{"path":"fixture.txt"}}</tool_call>',
            "tool_preamble": 'Inspect the file.\n</think>\n\nI will inspect the file now.\n' + self.xml,
            "nested_object": 'Inspect the file.\n</think>\n\n' + self.xml.replace('</function>', '<parameter=options>\n{"a":1,"b":2}\n</parameter>\n</function>'),
            "multiline_string": 'Inspect the file.\n</think>\n\n' + self.xml.replace('fixture.txt', 'line one\n  line two\nline three'),
        }
        for name, raw in cases.items():
            with self.subTest(name=name):
                self.cache = service.TokenHistory()
                messages, original = self.response(raw)
                followup = self.add_result(messages)
                reconstructed, preserved = self.continuation(followup)
                self.assertEqual(preserved, len(original))
                self.assertEqual(reconstructed[:len(original)], original)
                # This is the deployed engine's consumed-prefix match condition.
                self.assertEqual(reconstructed[:len(original)-1], original[:-1])
                suffix = self.render(followup)[len(self.render(messages, generation=False)):]
                self.assertEqual(reconstructed[len(original):], self.encode('\n' + suffix))

    def test_multiple_roundtrips_keep_earlier_raw_histories(self):
        messages = self.messages
        previous = []
        for index in range(4):
            raw = f'Inspect step {index}.\n\n</think>\n\n' + self.xml.replace('fixture.txt', f'fixture{index}.txt')
            messages, original = self.response(raw, messages)
            self.assertEqual(original[:len(previous)], previous)
            messages = self.add_result(messages)
            ids, preserved = self.continuation(messages)
            self.assertEqual(ids[:len(original)], original)
            self.assertEqual(preserved, len(original))
            previous = original

    def test_real_history_and_rendering_edits_do_not_reuse(self):
        messages, _ = self.response('Inspect it.\n\n</think>\n\n' + self.xml)
        followup = self.add_result(messages)
        for location in ['system', 'user', 'reasoning', 'argument', 'call_id']:
            changed = copy.deepcopy(followup)
            if location == 'system': changed[0]['content'] += ' Edited.'
            elif location == 'user': changed[1]['content'] += ' Edited.'
            elif location == 'reasoning': changed[2]['reasoning_content'] += ' Edited.'
            elif location == 'argument': changed[2]['tool_calls'][0]['function']['arguments'] = '{"path":"different.txt"}'
            else: changed[2]['tool_calls'][0]['id'] += '_edited'
            with self.subTest(location=location):
                ids, preserved = self.continuation(changed)
                self.assertEqual(preserved, 0)
                self.assertEqual(ids, self.encode(self.render(changed)))
        tools = copy.deepcopy(self.tools); tools[0]['function']['description'] = 'Changed tool.'
        self.assertEqual(self.continuation(followup, tools=tools)[1], 0)
        for effort in ['none', 'low', 'medium']:
            self.assertEqual(self.continuation(followup, opts=service.template_options({'reasoning_effort':effort}))[1], 0)

    def test_new_tool_result_is_appended_verbatim(self):
        messages, original = self.response('Inspect it.\n\n</think>\n\n' + self.xml)
        followup = self.add_result(messages);followup[-1]['content'] = 'Tool says <literal>\n  keep spacing\nnew result.'
        ids, preserved = self.continuation(followup)
        self.assertEqual(preserved, len(original))
        self.assertIn('Tool says <literal>\n  keep spacing\nnew result.', self.tok.decode(ids[len(original):]))

    def test_partial_response_closing_boundary(self):
        messages, original = self.response('Think.\n</think>\n\nA length-limited answer', terminated=False)
        followup = messages + [{'role':'user','content':'Continue.'}]
        ids, preserved = self.continuation(followup)
        self.assertEqual(preserved,len(original));self.assertEqual(ids[:len(original)],original)
        self.assertTrue(self.tok.decode(ids[len(original):]).startswith('<|im_end|>\n'))

    def test_no_thinking_history_is_preserved(self):
        self.opts = service.template_options({'reasoning_effort':'none'})
        messages = service.prepare_messages(self.messages);prompt = self.render(messages);ids = self.encode(prompt)
        raw = ' A reply with its original trailing space.  ';out = self.encode(raw+'<|im_end|>')
        assistant={'role':'assistant','content':raw}
        self.cache.remember(self.tok,messages,self.tools,self.opts,ids,out,self.eos,assistant)
        new, preserved=self.continuation(messages+[assistant,{'role':'user','content':'Next.'}])
        self.assertEqual(preserved,len(ids)+len(out));self.assertEqual(new[:preserved],ids+out)

    def test_context_dependent_template_rewrite_falls_back(self):
        self.opts['preserve_thinking'] = False
        # This older assistant loses its reasoning once the last user index advances.
        self.messages.insert(1, {'role':'user','content':'An earlier question.'})
        self.messages.insert(2, {'role':'assistant','content':'Earlier answer.','reasoning_content':'Older thought.'})
        messages, _ = self.response('Inspect it.\n</think>\n\n'+self.xml)
        followup=self.add_result(messages)+[{'role':'user','content':'A new user turn.'}]
        ids,preserved=self.continuation(followup)
        self.assertEqual(preserved,0);self.assertEqual(ids,self.encode(self.render(followup)))

    def test_interchangeable_api_forms_and_argument_key_order(self):
        raw='Inspect it.\n\n</think>\n\n'+self.xml.replace('</function>','<parameter=options>\n{"a":1,"b":2}\n</parameter>\n</function>')
        messages, original=self.response(raw)
        followup=self.add_result(messages);followup[-2]['content']=''
        args=json.loads(followup[-2]['tool_calls'][0]['function']['arguments'])
        args['options']={'b':2,'a':1};followup[-2]['tool_calls'][0]['function']['arguments']=json.dumps(args,separators=(',',':'))
        ids,preserved=self.continuation(followup,opts=dict(reversed(list(self.opts.items()))))
        self.assertEqual(preserved,len(original));self.assertEqual(ids[:preserved],original)

    def test_javascript_numeric_and_empty_thinking_roundtrip(self):
        raw='\n\n</think>\n\n'+self.xml.replace('</function>','<parameter=options>\n{"a":1.0,"b":-0.0}\n</parameter>\n</function>')
        messages,original=self.response(raw)
        followup=self.add_result(messages);followup[-2].pop('reasoning_content',None)
        followup[-2]['tool_calls'][0]['function']['arguments']='{"path":"fixture.txt","options":{"a":1,"b":0}}'
        ids,preserved=self.continuation(followup)
        self.assertEqual(preserved,len(original));self.assertEqual(ids[:preserved],original)

    def test_bounded_entries_and_payload(self):
        self.cache=service.TokenHistory(max_entries=2,max_bytes=1000000)
        for index in range(3):
            self.response(f'Think.\n</think>\n\nAnswer {index}.')
        self.assertEqual(self.cache.stats()['entries'],2)
        self.assertLessEqual(self.cache.stats()['payload_bytes'],1000000)
        self.cache=service.TokenHistory(max_bytes=1)
        ids=self.encode(self.render(self.messages))
        self.assertFalse(self.cache.remember(self.tok,self.messages,self.tools,self.opts,ids,[1],self.eos,{'role':'assistant','content':'answer'}))
        self.assertEqual(self.cache.stats()['entries'],0)


if __name__ == '__main__':
    unittest.main()
