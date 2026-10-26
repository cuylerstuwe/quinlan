"""Media identities, original-token replay, and independent position properties."""
import base64
from importlib.util import find_spec
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from quinlan.vision import image_bytes, visual_ids, position_ids, VisionTokenizer, VisionEngineMixin
from quinlan import server


class VisionTests(unittest.TestCase):
    def test_data_url_validation_and_no_url_fetch(self):
        self.assertEqual(image_bytes('data:image/png;base64,' + base64.b64encode(b'fixture').decode()), b'fixture')
        for value in ['file:///fixture.png', 'http://127.0.0.1/image', 'https://example.com/image',
                      'data:image/png;base64,!', 'data:image/png,raw', 'data:image/png;base64,']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_bytes(value)
        with self.assertRaises(ValueError):
            image_bytes('data:image/png;base64,' + base64.b64encode(b'12345').decode(), max_bytes=4)

    def test_replayed_visual_rows_keep_identical_ids(self):
        self.assertEqual(visual_ids([1, 9, 9, 2, 9], 9, 3), [1, -1, -2, 2, -3])
        self.assertEqual(visual_ids([1, -1, -2, 2, 9], 9, 3), [1, -1, -2, 2, -3])
        with self.assertRaises(ValueError):
            visual_ids([1, 9], 9, 2)

    def test_prefix_template_expansion(self):
        base = Mock()
        base.apply_chat_template.return_value = 'A<|image_pad|>B<|image_pad|>C'
        tok = VisionTokenizer(base, [4, 8])
        full = tok.apply_chat_template([], tokenize=False)
        self.assertEqual(full.count('<|image_pad|>'), 12)
        base.apply_chat_template.return_value = 'A<|image_pad|>B'
        self.assertTrue(full.startswith(tok.apply_chat_template([], tokenize=False)))
        base.apply_chat_template.return_value = '<|image_pad|>' * 3
        with self.assertRaises(ValueError):
            tok.apply_chat_template([], tokenize=False)

    @unittest.skipUnless(find_spec('numpy'), 'numpy is part of inference extra')
    def test_image_positions_and_post_image_text(self):
        ids = [1, 2, -1, -2, -3, -4, -5, -6, 3, 4]
        positions, delta = position_ids(ids, [(1, 4, 6)])
        self.assertEqual(positions.tolist(), [[0,0,0],[1,1,1], [2,2,2],[2,2,3],[2,2,4],
            [2,3,2],[2,3,3],[2,3,4],[5,5,5],[6,6,6]])
        self.assertEqual(delta, -3)
        with self.assertRaises(ValueError):
            position_ids(ids, [(2, 4, 6)])

    def test_admission_never_reconfigures_an_active_request(self):
        engine = SimpleNamespace(vision=Mock(), active={0: object()}, prefills=[], q=[object()])
        VisionEngineMixin._admit(engine)
        engine.vision.activate.assert_not_called()

    def test_append_keeps_existing_image_prefix_but_replacement_evicts(self):
        class Base:
            def _admit(self):
                self.admitted = True
        class Engine(VisionEngineMixin, Base):
            pass
        for previous, current, evicted in [(('red',),('red','blue'),False),
                (None,('red',),False), (('red',),('blue',),True),
                (('red','blue'),('blue','red'),True), (('red',),None,True)]:
            eng = Engine()
            eng.active, eng.prefills, eng.parked = {}, [], {0:object()}
            eng.host_parked, eng.host_bytes = {}, 0
            eng.vision = SimpleNamespace(active_key=previous,last_metrics={},activate=Mock())
            eng.q = [SimpleNamespace(input_state=SimpleNamespace(key=current) if current else None)]
            eng._evict = Mock()
            eng._admit()
            self.assertEqual(eng._evict.called,evicted)
            self.assertTrue(eng.admitted)


@unittest.skipUnless(os.environ.get('QWENTIN_MODEL_DIR'), 'requires the real tokenizer')
class ImageHistoryTests(unittest.TestCase):
    def test_image_tool_roundtrip_preserves_original_tokens(self):
        from transformers import AutoTokenizer
        base = AutoTokenizer.from_pretrained(os.environ['QWENTIN_MODEL_DIR'], local_files_only=True)
        tok = VisionTokenizer(base, [64])
        cache = server.TokenHistory()
        opts = server.template_options({'reasoning_effort': 'none'})
        messages = [{'role':'user','content':[{'type':'image','image':'synthetic-image-digest'},
                                             {'type':'text','text':'Inspect the image.'}]}]
        render = lambda ms, gen: tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=gen, **opts)
        prompt = render(messages, True)
        ids, _ = cache.resolve(tok, messages, None, opts, prompt)
        ids = visual_ids(ids, base.convert_tokens_to_ids('<|image_pad|>'), 64)
        out = base('red<|im_end|>', add_special_tokens=False).input_ids
        assistant = {'role':'assistant','content':'red'}
        self.assertTrue(cache.remember(tok, messages, None, opts, ids, out, {out[-1]}, assistant))
        continued = messages + [assistant, {'role':'user','content':'Say the color again.'}]
        reconstructed, preserved = cache.resolve(tok, continued, None, opts, render(continued, True))
        self.assertEqual(reconstructed[:len(ids)+len(out)], ids + out)
        self.assertEqual(preserved, len(ids) + len(out))
        positions, delta = position_ids(reconstructed, [(1,16,16)])
        self.assertLess(delta, 0)
        self.assertEqual(len(positions), len(reconstructed))
        edited = [{'role':'user','content':[{'type':'image','image':'different-digest'},
            {'type':'text','text':'Inspect the image.'}]}] + continued[1:]
        self.assertEqual(cache.resolve(tok, edited, None, opts, render(edited, True))[1], 0)
