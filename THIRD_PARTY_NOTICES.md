# Third-party notices

This work is distributed under GNU Affero General Public License version 3.
The complete license is in `LICENSE`. Original notices and credits are retained.
No official affiliation with Forge Neo, ComfyUI, AUTOMATIC1111 or model authors is implied.

## Forge Neo / ComfyUI / k-diffusion

Reference: Haoming02/sd-webui-forge-classic,
commit `710f1e25fcac84d880cbccf27d11b2e3276e589e` (AGPL-3.0).

Files in `vendor/forge/` are extracted from the pinned Forge source. Sampling
arithmetic originates in part from ComfyUI and k-diffusion by Katherine Crowson
and their respective contributors. See headers and `vendor/source_manifest.json`
for exact paths and original-source SHA256s.

Changes: retain only the V1 sampler/planner subset; explicit per-run noise
injection; no process-wide TorchHijack; require a private Brownian sampler;
use Forge's actual scalar-sigma `to_d` override; package-local Lark import.
The prompt grammar and timestep planning are retained from the reference.
The sampler subset now also includes LCM, LMS, DPM++ SDE, DPM++ 3M SDE and
Res Multistep. Res Multistep is specialized to its registered eta=0/CFG++-off
entry; LCM and Res Multistep receive the private per-run RNG. LMS reuses the
host's existing SciPy integration. The source SHA in the manifest is unchanged.

`bridge/schedules.py`, `bridge/rng.py`, `bridge/classic.py`, and the relevant
conditioning/CFG adapters in `bridge/text.py` and `bridge/runtime.py` contain
adaptations of Forge's corresponding algorithms. They are not independent
claims to authorship of the upstream algorithms.

## ComfyUI-Neo-Sampler

Reference: wangjue520/ComfyUI-Neo-Sampler,
commit `925257cf3a97ea203d30d37fb24b3657721c986a` (AGPL-3.0).

Its RNG/configuration and text-engine bridging architecture informed and were
adapted into Bridge. This project does not import that extension at runtime.
Bridge removes shared/global RNG writes, mutable scheduler context, global
torch proxies, startup pip installation, and automatic OOM tiling. Its tests
use pinned Forge as the oracle, not the other project's README results.

## Lark 1.3.1 — MIT

A private source copy is in `vendor/lark/`; the full original copyright and MIT
license are in `vendor/lark/LICENSE`.
Upstream: https://github.com/lark-parser/lark

Changes are package-local import paths and namespace/logger/resource lookup
needed to avoid replacing the user's global `lark` package. Unused command-line
tools and packaging hooks were omitted. Reproduction is documented by
`tools/vendor_lark.py`; it is developer-only and never called on node import.

## Existing host dependencies

PyTorch, NumPy, SciPy, Pillow, tqdm and ComfyUI are supplied by the host.
DPM++ 2M SDE uses the host's torchsde 0.2.6 (Apache-2.0) implementation in a
per-run private namespace. The installed module is not modified. Source is
read only from the trusted installed dependency, never from image metadata.
No third-party model weights or user-provided image/prompt fixtures are included.
