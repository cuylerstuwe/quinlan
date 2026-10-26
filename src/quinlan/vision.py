"""Optional BF16 image encoder with request-local native embeddings and M-RoPE.

Accepts OpenAI image_url parts containing base64 data URLs. Network URLs, video,
and audio are rejected explicitly; the server never fetches client-supplied URLs.
"""
import base64
import copy
import ctypes
from dataclasses import dataclass
import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path
import threading
import time


@dataclass
class ImageInput:
    key: str
    grid: tuple
    pixels: object
    features: object = None


@dataclass
class VisionInput:
    images: list
    positions: object = None
    delta: int = 0

    @property
    def key(self):
        return tuple(image.key for image in self.images)

    @property
    def tokens(self):
        return sum(t * h * w // 4 for t, h, w in (image.grid for image in self.images))


def image_bytes(url, max_bytes=20 * 2**20):
    if not isinstance(url, str) or not url.startswith('data:image/'):
        raise ValueError('Images require a base64 data:image/... URL; network/file URLs are not fetched')
    header, sep, encoded = url.partition(',')
    if not sep or not header.endswith(';base64') or len(encoded) > (max_bytes + 2) // 3 * 4:
        raise ValueError('Invalid or oversized base64 image')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid base64 image') from exc
    if not raw or len(raw) > max_bytes:
        raise ValueError('Invalid or oversized image')
    return raw


class VisionTokenizer:
    """Use the unchanged chat template and expand only its image placeholders.

    Prefix rendering inside TokenHistory sees the same expansion, preserving
    the original-token round-trip fix for conversations containing images.
    """
    def __init__(self, tokenizer, counts):
        self.base = tokenizer
        self.counts = counts

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        if kwargs.get('tokenize', True):
            raise ValueError('VisionTokenizer renders text; tokenize the expanded result separately')
        rendered = self.base.apply_chat_template(*args, **kwargs)
        parts = rendered.split('<|image_pad|>')
        count = len(parts) - 1
        if count > len(self.counts):
            raise ValueError('Image placeholders do not match the image content')
        return ''.join(part + ('<|image_pad|>' * self.counts[i] if i < count else '')
                       for i, part in enumerate(parts))


def visual_ids(ids, image_token_id, count):
    """Project positive image placeholders and replayed negative IDs identically."""
    result, row = [], 0
    for token in ids:
        if token == image_token_id or token < 0:
            result.append(-row - 1)
            row += 1
        else:
            result.append(token)
    if row != count:
        raise ValueError('Image token count does not match encoder output')
    return result


def position_ids(ids, grids):
    """Qwen3.5/3.8 image positions, independently checked against Transformers."""
    import numpy as np
    positions = np.empty((len(ids), 3), dtype=np.int32)
    index = current = image_index = 0
    while index < len(ids):
        if ids[index] >= 0:
            end = index + 1
            while end < len(ids) and ids[end] >= 0:
                end += 1
            positions[index:end] = (np.arange(end - index, dtype=np.int32) + current)[:, None]
            current += end - index
            index = end
        else:
            if image_index >= len(grids):
                raise ValueError('Unexpected visual token run')
            t, h, w = grids[image_index]
            if t != 1 or h % 2 or w % 2:
                raise ValueError('Only still-image grids with spatial merge size 2 are supported')
            h, w = h // 2, w // 2
            length = h * w
            if index + length > len(ids) or any(x >= 0 for x in ids[index:index + length]):
                raise ValueError('Image grid and token run disagree')
            positions[index:index + length, 0] = current
            positions[index:index + length, 1] = np.repeat(np.arange(h), w) + current
            positions[index:index + length, 2] = np.tile(np.arange(w), h) + current
            current += max(h, w)
            index += length
            image_index += 1
    if image_index != len(grids):
        raise ValueError('Missing image token run')
    return positions, current - len(ids)


class VisionTower:
    def __init__(self, directory, lib, *, expected_config=None, max_pixels=4194304, max_tokens=8192, cache_bytes=128 * 2**20):
        import torch
        from safetensors import safe_open
        from transformers import initialization
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        self.torch, self.lib = torch, lib
        directory = Path(directory)
        full_config = json.loads((directory / 'config.json').read_text())
        if expected_config is not None and full_config['vision_config'] != expected_config:
            raise ValueError('Vision configuration differs from the text checkpoint')
        config = Qwen3_5VisionConfig(**full_config['vision_config'])
        if config.spatial_merge_size != 2 or config.out_hidden_size != 5120:
            raise ValueError('Unsupported vision architecture')
        config._attn_implementation = 'sdpa'
        # CPU BF16 construction avoids allocating a second random FP32 tower.
        with torch.device('meta'), initialization.no_init_weights():
            self.model = Qwen3_5VisionModel(config)
        self.model.to_empty(device='cpu')
        # Nonpersistent rotary buffers must be reconstructed after to_empty.
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
        self.model.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config)
        state = {}
        for file in sorted(directory.glob('*.safetensors')):
            with safe_open(file, framework='pt', device='cpu') as source:
                for name in source.keys():
                    if name.startswith('model.visual.'):
                        tensor = source.get_tensor(name)
                        if tensor.dtype != torch.bfloat16:
                            raise ValueError('The vision tower requires unquantized BF16 tensors')
                        state[name.removeprefix('model.visual.')] = tensor
        if not state:
            raise ValueError('No model.visual.* tensors found; retain the vision component when quantizing')
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.eval().requires_grad_(False).to('cuda')
        self.processor = Qwen2VLImageProcessorPil.from_pretrained(directory, local_files_only=True)
        self.max_pixels, self.max_tokens = max_pixels, max_tokens
        self.image_token_id = full_config['image_token_id']
        self.cache, self.cache_bytes, self.cache_used = OrderedDict(), cache_bytes, 0
        self.lock = threading.Lock()
        self.active_key, self.active_embeddings = None, None
        self.last_metrics = {}
        lib.quinlan_vision_configure.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.quinlan_vision_configure.restype = ctypes.c_int

    def prepare(self, messages, tokenizer):
        from PIL import Image, ImageOps, UnidentifiedImageError
        images, counts = [], []
        normalized = copy.deepcopy(messages)
        for message in normalized:
            content = message.get('content')
            if isinstance(content, str):
                if any(x in content for x in ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>')):
                    raise ValueError('Literal vision control tokens are reserved')
                continue
            if content is None:
                continue
            if not isinstance(content, list):
                raise ValueError('Message content must be text or a content-part array')
            for index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise ValueError('Invalid message content part')
                if part.get('type') == 'text':
                    if not isinstance(part.get('text'), str) or any(x in part['text'] for x in ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>')):
                        raise ValueError('Invalid text content or reserved vision control token')
                    continue
                if part.get('type') != 'image_url' or message.get('role') not in ('user', 'tool'):
                    raise ValueError('Supported media: user/tool image_url parts; video and audio are not supported')
                info = part.get('image_url')
                url = info.get('url') if isinstance(info, dict) else info
                raw = image_bytes(url)
                key = hashlib.sha256(raw).hexdigest()
                with self.lock:
                    cached = self.cache.get(key)
                if cached:
                    grid, pixels = cached[0], None
                else:
                    try:
                        with Image.open(io.BytesIO(raw)) as image:
                            if image.width * image.height > 33554432 or getattr(image, 'n_frames', 1) != 1:
                                raise ValueError('Image is too large or animated')
                            image = ImageOps.exif_transpose(image).convert('RGB')
                            batch = self.processor(images=image, return_tensors='pt',
                                size={'shortest_edge': 65536, 'longest_edge': self.max_pixels})
                    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                        raise ValueError('Invalid image data') from exc
                    grid = tuple(batch['image_grid_thw'][0].tolist())
                    pixels = batch['pixel_values']
                images.append(ImageInput(key, grid, pixels, cached[1] if cached else None))
                counts.append(grid[0] * grid[1] * grid[2] // 4)
                if len(images) > 8 or sum(counts) > self.max_tokens:
                    raise ValueError('At most 8 images and 8192 visual tokens are supported per request')
                content[index] = {'type': 'image', 'image': key}
        if not images:
            return normalized, tokenizer, None
        return normalized, VisionTokenizer(tokenizer, counts), VisionInput(images)

    def activate(self, state):
        """Only the native engine owner may touch GPU state or embedding cache."""
        torch = self.torch
        started = time.perf_counter()
        hits = 0
        if state is None:
            # Pure text runs exactly the existing native path: no new CUDA call.
            rc = self.lib.quinlan_vision_configure(None, 0, None, 0, 0) if self.active_key is not None else 0
            self.active_key, self.active_embeddings = None, None
        else:
            if self.active_key != state.key:
                features = []
                for image in state.images:
                    with self.lock:
                        cached = self.cache.get(image.key)
                        if cached:
                            self.cache.move_to_end(image.key)
                    if cached is None and image.features is not None:
                        cached = (image.grid, image.features)
                    if cached:
                        feature = cached[1].to('cuda')
                        hits += 1
                    else:
                        if image.pixels is None:
                            raise RuntimeError('Image cache changed during request preparation; resend the image')
                        with torch.inference_mode():
                            feature = self.model(image.pixels.to(device='cuda', dtype=torch.bfloat16),
                                grid_thw=torch.tensor([image.grid], device='cuda')).pooler_output.contiguous()
                        cpu_feature = feature.cpu()
                        size = cpu_feature.numel() * cpu_feature.element_size()
                        with self.lock:
                            old = self.cache.pop(image.key, None)
                            if old:
                                self.cache_used -= old[1].numel() * old[1].element_size()
                            self.cache[image.key] = (image.grid, cpu_feature)
                            self.cache_used += size
                            while self.cache_used > self.cache_bytes and self.cache:
                                _, removed = self.cache.popitem(last=False)
                                self.cache_used -= removed[1].numel() * removed[1].element_size()
                    features.append(feature)
                self.active_embeddings = torch.cat(features).contiguous()
                self.active_key = state.key
            else:
                hits = len(state.images)
            torch.cuda.synchronize()
            rc = self.lib.quinlan_vision_configure(self.active_embeddings.data_ptr(), state.tokens,
                state.positions.ctypes.data, len(state.positions), state.delta)
        if rc:
            raise RuntimeError(f'Native image configuration failed ({rc})')
        self.last_metrics = {'images': len(state.images) if state else 0,
            'image_tokens': state.tokens if state else 0, 'embedding_cache_hits': hits,
            'prepare_ms': (time.perf_counter() - started) * 1000}


class VisionEngineMixin:
    """Configure single-slot visual state before prefix matching, on its owner."""
    def _admit(self):
        tower = getattr(self, 'vision', None)
        if tower is None:
            return super()._admit()
        if self.active or self.prefills or not self.q:
            return
        req = self.q[0]
        state = getattr(req, 'input_state', None)
        key = state.key if state else None
        try:
            old_images, new_images = tower.active_key or (), key or ()
            if new_images[:len(old_images)] != old_images:
                # Internal negative row IDs alone cannot identify image content.
                # Appending new images preserves earlier row identities and
                # positions. Replacing/removing/reordering old images does not.
                for slot in list(self.parked):
                    self._evict(slot)
                self.host_parked.clear()
                self.host_bytes = 0
            tower.activate(state)
            req.vision_metrics = dict(tower.last_metrics)
        except Exception:
            self.q.pop(0)
            self._finish(req, 'Image preparation failed')
            return
        return super()._admit()
