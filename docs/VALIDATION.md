# Validation status — implementation 1.0.0

The approved `SPEC_JA.md` remains the design contract. This document records
implementation testing; it does not convert the formal GPU matrix to PASS.

## Executed locally

| Check | Result | Scope |
|---|---|---|
| Python pytest | 311 PASS | Spec, metadata, token/CFG contracts, binding, RNG, immutable replay, pinned CPU oracle |
| Node.js tests | 14 PASS | Detached graph construction, named ports/widgets, exact seed text, native priority/delegation |
| Actual Comfy CPU integration | 9 PASS | 3 families × Euler, ER-SDE, DPM++ 2M SDE; analytic diffusion models; real ModelPatcher/CFGGuider |
| Actual Comfy text API | 3 PASS | Real CLIP/Qwen transformer classes with tiny synthetic weights/tokenizers |
| Injected interruption/retry | 3 PASS | One per family in the real Core CPU smoke; same output on retry |
| Chromium editor harness | 8 checks PASS | Real Bridge DOM and backend helpers; Comfy host mocked |
| Supplied WebP metadata | 6 files checked | 3 Forge parameters extracted; 3 native workflows prioritized; private images not published |

The pinned oracle tests compare initial and later noise, sigma vectors, sampling
arithmetic and denoiser call counts. They use analytic denoisers, not pretrained
model weights. The reported counts are distinct test cases, not 311 generated
images and not the 44 formal qualification definitions.

Local environment: Python 3.13.5, PyTorch 2.10.0+cpu, torchsde 0.2.6.
Comfy source: `88ab4a06566454ad89db8f0bedb970d6c08cd1b7`.
Forge oracle: `710f1e25fcac84d880cbccf27d11b2e3276e589e`.
No GPU was present. Additional host dependencies were placed only in an isolated
test directory. No user's ComfyUI installation was changed.

## Not performed / no qualification claim

- Actual Anima-Luc/Turbo, SD1.5 or SDXL pretrained-model GPU A/B.
- Exact final image/VAE parity, quantized weights, arbitrary attention backends.
- Full live ComfyUI frontend + actual loader file selections.
- H3/LTX native image/video/audio workflows, queues and GPU memory endurance.
- Cross-machine or all-model compatibility.

The real Core smoke uses synthetic diffusion/adapter weights and manually built
synthetic conditioning. The text API check uses small configurations. The browser
harness evaluates the real editor in Chromium without network navigation and
mocks only the Comfy host/asset catalog. These are API/lifecycle tests, not a
substitute for the missing full GUI or real-model GPU gates.

`certification.json` has no qualified profiles. All runtime Reports keep
`qualification=not_evaluated` until an independent comparison is supplied.

## Reproduction

1. Use a clean development environment, not the user's production Comfy env.
2. Obtain the exact Forge source and set `FORGE_REFERENCE` to its root.
3. Run `python -m pytest -q` and `node --test tests/frontend.test.mjs`.
4. Install the pinned Comfy dependencies in that development environment and run
   `python tools/core_smoke.py /path/to/ComfyUI` with the pinned source.
5. Run `python tools/browser_smoke.py` with developer-installed Playwright and
   Chromium (`CHROMIUM` may override `/usr/bin/chromium`).
6. Follow `test_matrix.json` for real-model/GPU qualification. Register complete
   model/TE/VAE/LoRA hashes, source versions and numeric/attention configuration.

The repository CI performs CPU/Node checks on Ubuntu and Windows; the additional
Core/browser harness is run on Ubuntu. Consult the actual Actions result for the
published commit; a skipped oracle or optional test is not a PASS.

## Non-interference design

No runtime writes to Comfy sampler functions, torch default generators, shared
scheduler context or third-party module globals are used. Consumers clone tensor
entries and reconstruct per-run RNGs from byte snapshots. The Brownian adapter
owns and releases its interval tree. OOM is propagated; no automatic tiling,
precision, resolution or step changes are made.

These properties have CPU/import/replay/exception tests. H3/LTX end-to-end
non-interference still requires the separate formal GPU gate in the specification.
