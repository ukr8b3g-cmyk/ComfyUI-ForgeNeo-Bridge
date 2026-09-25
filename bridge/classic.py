# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from wangjue520/ComfyUI-Neo-Sampler 925257cf (Forge 710f1e25 oracle).
# Changes: run-local backtrack, explicit dtype, no CLIP option mutations.
import torch
from ..vendor.forge.parsing import parse_prompt_attention


def apply_emphasis(name, z, multipliers):
    """backend/text_processing/emphasis.py"""
    if name == "Original":
        original_mean = z.mean()
        z = z * multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)
    elif name == "No norm":
        z = z * multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)
    # "Ignore" and "None": after_transformers() does nothing
    return z

class _Chunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []

class ClassicEngine:
    """backend/text_processing/classic_engine.py (textual inversion not supported)"""

    def __init__(self, sdclip, hf_tokenizer, emphasis, chunk_length=75, text_projection=False, minimal_clip_skip=1, clip_skip=1, return_pooled=False, final_layer_norm=True, comma_padding_backtrack=20):
        self.sdclip = sdclip  # comfy.sd1_clip.SDClipModel
        self.tokenizer = hf_tokenizer
        self.emphasis = emphasis
        self.chunk_length = chunk_length
        self.text_projection = text_projection
        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm
        self.comma_padding_backtrack = comma_padding_backtrack

        special = getattr(sdclip, "special_tokens", {})
        self.id_start = special.get("start", 49406)
        self.id_end = special.get("end", 49407)
        self.id_pad = special.get("pad", 49407)  # Neo: tokenizer.pad_token_id (clip_l: 49407, SDXL clip_g: 0 "!")

        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab[",</w>"]

    def empty_chunk(self):
        chunk = _Chunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def tokenize(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def tokenize_line(self, line):
        parsed = parse_prompt_attention(line, self.emphasis)
        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = _Chunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = _Chunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                comma_padding_backtrack = self.comma_padding_backtrack

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        self.last_token_trace = {"ids": [c.tokens for c in chunks], "weights": [c.multipliers for c in chunks]}
        return chunks, token_count

    def encode_with_transformers(self, tokens, device):
        transformer = self.sdclip.transformer  # comfy.clip_model.CLIPTextModel
        tokens = tokens.to(device)
        layer_id = -max(self.clip_skip, self.minimal_clip_skip)
        # Neo: hidden_states[layer_id] (+ final_layer_norm), computed with float32 embeddings
        out = transformer(tokens, None, intermediate_output=layer_id, final_layer_norm_intermediate=self.final_layer_norm, dtype=torch.float32)
        z = out[1]
        pooled = None
        if self.return_pooled:
            pooled = out[2] if self.text_projection else out[3]
        return z, pooled

    def process_tokens(self, remade_batch_tokens, batch_multipliers, device):
        tokens = torch.asarray(remade_batch_tokens)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1 : tokens.shape[1]] = self.id_pad

        z, pooled = self.encode_with_transformers(tokens, device)
        multipliers = torch.asarray(batch_multipliers).to(z)
        z = apply_emphasis(self.emphasis, z, multipliers)
        return z, pooled

    def __call__(self, line, device):
        chunks, _ = self.tokenize_line(line)
        zs = []
        first_pooled = None
        for i, chunk in enumerate(chunks):
            z, pooled = self.process_tokens([chunk.tokens], [chunk.multipliers], device)
            if i == 0:
                first_pooled = pooled
            zs.append(z)
        z = torch.hstack(zs)
        return (z, first_pooled) if self.return_pooled else z