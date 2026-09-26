# Validation status — implementation 1.0.0

## Deep-audit fixes — 2026-09-27

- Reviewed the supplied diagnostic ZIP against current code and reproduced the relevant paths. CFG++ now retains negative conditioning at CFG 1, including an empty negative prompt. Core-backed Euler CFG++, Euler a CFG++ and DPM++ 2M CFG++ were checked with actual ComfyUI CPU sampling and both adjustment modes. Skip Early CFG and NGMS callback cases are covered by focused tests.
- LoRA graph reconstruction now retains model-only and text-only applications, repeated applications of the same file, and known shared-loader order. Qwen Edit 1.x/2509/2511 reference workflows connect both positive and negative Edit conditioning to the same source image and VAE. These are graph/API checks, not pretrained-image parity tests.
- A Forge `Shift: 0` sentinel is retained in the source request and mapped to the native Z-Image/ERNIE default Shift 3.0. A supplied Z-Image WebP that previously failed now produces a ten-node graph. Its `pig_flux_vae_fp32-f16` VAE is absent from the current registered catalog; generation still requires selecting a suitable installed VAE.
- Seed bounds check the offsets actually used by variation, batch and Sampling adjustments. Disabled ENSD is retained but does not reject or perturb a valid seed. Summary traces omit per-call/per-step tensor hashes while retaining a call count and final integrity hashes; detailed tensor tracing remains available. A degenerate all-zero synthetic text-test fixture was corrected without changing production emphasis behavior.
- New arrangement frames use serialized ownership and spatial membership. Isolated real-ComfyUI frontend checks confirmed save/reload, copied frames with remapped IDs, partial selection and manual-frame protection, with API inputs unchanged. Unmarked legacy frames are deliberately retained and may overlap after the first rearrangement.
- Validation: **610 Python tests passed** against the pinned Forge reference, **51 frontend tests passed**, 27 actual-core CPU analytic/synthetic-text cases, 60 shared-sampler CPU cases and 204 unadjusted CPU cases passed. Separate CFG 1 core runs showed zero latent difference against the restored-negative control and exact core agreement with adjustments OFF. Ten supplied Forge WebPs built schema-valid graphs against the installed node definitions and model catalog. No new pretrained GPU run or restart of the user's ComfyUI server was performed.

## Illustrated documentation and publication checks — 2026-09-26

- Full Python suite against a freshly retrieved pinned Forge `710f1e25` reference: **482 passed, 0 skipped**. The installed Forge checkout had advanced to `3a3dea7f`, so it was not used as the fixed oracle. PyTorch nested-tensor prototype and torchsde floating-point boundary warnings remain (45 warnings); no test failures.
- All five frontend test files: **49 passed**. CI now runs all five rather than only the original frontend test file.
- Japanese-first / English-second README: 63 local/remote/anchor link occurrences checked, plus 14 in the retained technical guide; all relative targets exist. GitHub's Markdown API preserves both language anchors, the comparison table with 320px display widths, section images and localized video links. Cropped controls were visually inspected. Both comparison originals remain 2048×2688 with metadata intact. These documentation checks do not add GPU or browser-interaction coverage.
- The user confirmed Anima Hires.fix operation after the previous correction. Documented that Hires.fix may differ slightly, and in some conditions more substantially, from Forge output; neither exact nor visual parity is certified.

## Anima Hires.fix reconstruction — 2026-09-26

- Fixed the SD1.5/SDXL-only Hires recipe gate for Anima and added second-pass Hires Shift. Unsupported Hires recipes now fail with a specific `HIRES_UNSUPPORTED` message instead of placing a misleading one-pass txt2img graph. Unrelated unsupported operations remain blocked.
- The image reconstruct endpoint reads container dimensions without decoding pixels. Explicit Hires resize takes precedence; otherwise image dimensions are used for scale-based Hires, with a note when they differ from multiplication. Text-only reconstruction records the multiplied dimensions as an estimate. Resized images therefore change the reconstructed target; use the original file.
- Focused regression: **100 passed** across Hires, metadata, reconstruction and sampling-adjustment tests, including existing SD1.5/SDXL, independently different base/Hires Shift, explicit-size priority, unknown settings, missing upscaler and PNG/JPEG/WebP dimensions.
- Actual reconstruct endpoint handler called in-process with the supplied `20260926212845-waiANIMA_v10Base10-1153657013-Euler a.webp`: HTTP response 200, 15 nodes, 30/20 steps, CFG 5/4, denoise 1/0.5, Lanczos target 1536×2048. The old multiplication alone gave 1536×2016. The user's running server was not reloaded for this check.
- Isolated browser using the installed real frontend and the staged backend plan: two samplers, widget values, target dimensions and save/reload passed; no page errors. This was graph construction using the importer helper, not a successful end-to-end drop against the old live backend. Initial direct handleFile attempts did not place the graph and are not counted as passes. Evidence: `anima-hires-browser-result.json`, `anima-hires-ui.png`, `ForgeNeo Bridge anima hires fixed.json` in local validation assets.
- Standalone actual-model GPU execution with the user's queue empty: `waiANIMA_v10Base10`, `qwen_3_06b_base`, `qwenimagevae_v10`, seed 1153657013, Euler ancestral / normal. **Sampling adjustments OFF passed in 191.50 s; ON passed in 147.19 s**, including decode/upscale/encode and the second sampler. Both produced finite tensors and 1536×2048 PNGs. Separate processes; no user-managed backend restart or queue submission. Timings are observations, not a controlled performance comparison.
- Outputs: `anima-hires-default_00001_.png`, `anima-hires-adjusted_00001_.png`; corresponding GPU reports and API prompts are retained in local validation assets. Default OFF output differs in composition/details from the supplied Forge source. No pixel parity or general model-quality claim.

## Sampler preview room, larger output images and group padding — 2026-09-26

- Based on the supplied `ForgeNeo Bridge anima (1).json`: sampler saved at 220×494, Settings below it and SaveImage at 510×650. Arrangement now reserves at least 500px height for an expanded sampler plus a 60px gap; uses the actual height if larger. Output images expand to at least 510×650 without shrinking larger sizes. Group padding is top 100px, sides/bottom 30px. Collapsed nodes keep their size and state.
- Frontend tests: **49 passed**. Isolated real ComfyUI browser with the supplied workflow confirmed unchanged API inputs, preserved existing sizes, 60px gap below the loaded 500px-high sampler, 270px reserved gap below a 290px-high sampler, output size 510×650, save/reload and one-step Undo/Redo. Preview screenshots use an existing user-provided image only; no GPU generation or user queue operation. Evidence: local validation `preview-spacing-result.json`, `preview-spacing.png`, `ForgeNeo Bridge anima spaced.json`.
- Known audit finding remains open: copied group IDs can leave duplicate frames or delete unrelated hand-made frames. This spacing change does not claim to fix group ownership after copy/paste. The README now records the limitation.

## Compact workflow layout and Settings — 2026-09-26

- Frontend tests: **47 passed**, including detached construction, native import delegation, target isolation, stable connection-order placement, non-overlap, repeated frame replacement, unchanged dimensions/values and nested accordion persistence/localization.
- Isolated Chromium against the installed ComfyUI frontend, with staged JS and the existing sampling-adjustments schema injected if missing: loaded the user's `ForgeNeo-Bridge_00005_.png` embedded workflow (13 nodes). Native saved positions remained unchanged on load. Real canvas and node context-menu actions worked; **one Ctrl+Z** restored all original positions and removed all new frames, and Ctrl+Y restored the arrangement. Save/reload, F5/reimport and repeated arrangement retained five frames. An unrelated workflow node remained untouched.
- Measured bounds (including the same view padding) **3210×1352 → 2340×712**. Node dimensions, core names, links and generated API `class_type`/`inputs` were unchanged. Japanese/English menus and group titles, actual pointer clicks on Advanced settings and warning open/close, and fresh detached reconstruction were checked. No page exceptions. Evidence: `ForgeNeo-Bridge-validation/layout-browser-result.json`, `layout-after-ja.png`, `layout-after-en.png`, and `ForgeNeo Bridge compact.json` in the local validation folder.
- UI-only changes; no new GPU generation or rendering-quality claim. Non-GET requests were blocked in the isolated browser. No user backend restart, dependency addition, commit, push or release. Reload the user's browser to activate the deployed JS.
- Deployment: 10 changed/new files hash-verified against the local stage. Repeated the browser checks with JS served directly by the installed backend (`servedRuntimeJS: true`), without staged JS substitution. Live `object_info` already includes Sampling adjustments, so the optional schema fallback was not used in this run.

## Sampling adjustments and bilingual hover help — 2026-09-26

- Independent default-OFF Sampling adjustments under the existing default-ON Forge compatibility switch. Newly dropped workflows explicitly request OFF. Saved workflows predating the field migrate to ON to preserve previous behavior; explicit new choices round-trip. Linked Hires settings synchronize the matching switch only.
- Adjustments OFF uses actual core KSAMPLER and KSampler sigma generation with Forge text/initial noise retained. Adjustments ON retains the existing Forge path. Disabled sampling controls remain stored and are listed in `not_applied`. No global sampler/RNG patching; avoid the unused DDIM inpaint RNG reset because this path has no masks.
- Focused Python tests, including pinned Forge references, contracts, Hires, reconstruction, mode and adjustment coverage: **413 passed**. Frontend regression and bilingual help coverage: **40 passed**. Native locale files cover all **8 Bridge nodes and 77 visible inputs**, plus every output and node description. Accordion headers and seed controls use localized native widget tooltips.
- `tools/core_smoke.py ... --unadjusted`: **204 actual Comfy CPU cases passed** (17 samplers x 3 families x CFG 1/2 x txt2img/img2img). Compared against core CFGGuider with identical synthetic conditioning, initial noise and sigma vectors at rtol/atol 1e-6. Also checked repeatability, finite results and unchanged global RNG/model options/core functions. Both sides use inference mode, required for Anima's real encoder adapter. `--zero-denoise`: 3 additional family cases passed, returning the unchanged input latent.
- Isolated browser against the installed frontend, with staged extension files, node schema and translations substituted: Japanese/English field hover, output port and node-title help, 300px width, legacy ON/new OFF, API values after save/F5 and no page exceptions confirmed. Non-GET requests were blocked so this did not queue jobs or persist user settings. ComfyUI rebuilds the graph asynchronously when switching locale; checks wait for that rebuild before hovering.
- Deployed files are hash-checked against the stage. The user-managed backend is not restarted; restart ComfyUI and reload the browser to activate the new Python input. No new pretrained GPU test, image-similarity proof, commit, push or release.

## Named solver compatibility and README inventory — 2026-09-26

- Compare pinned Forge `710f1e25` with installed Comfy `830232b8`. README now separates common formulas, spelling aliases, demonstrated implementation differences, and unverified equivalence. Normal/DDIM endpoint handling, KL Optimal's different sigma vector, UniPC's terminal threshold, and AYS node differences are documented.
- Keep the existing 12 Forge solvers. Add installed Comfy solver calls for DDIM, UniPC, Euler CFG++, Euler a CFG++, and DPM++ 2M CFG++. Forge text, initial RNG, scheduling and exact-step Hires remain active. The ancestral CFG++ solver uses private continued RNG/ENSD; local post-CFG callbacks receive the negative prediction even at CFG 1. DDIM Eta/Churn remain unapplied and reported. No global core functions/options are patched.
- Focused spec/shared-solver/mode/reconstruction suite: 108 passed. An old spec test asserted DDIM must be rejected; change that negative fixture to still-unsupported PLMS and separately require all five new selections to execute while asset validation remains active. Related contracts passed in the earlier 91-test run. Pinned reference and Hires comparisons passed; the old DDIM expectation was the only failure in that 312-case run and is covered by the corrected focused run.
- `tools/core_smoke.py ... --shared-only`: 60 actual Comfy CPU cases passed (5 solvers x SD1.5/SDXL/Anima x CFG 1/2 x txt2img/exact-step img2img). Checks cover finite outputs, repeatability, unchanged global RNG, model options/object patches and core functions. `--sampling-only`: the existing 24 solver/model cases also completed without assertion failures. These use synthetic models and conditioning, not pretrained image quality.
- No new GPU generation or Forge-vs-Comfy numerical parity test for the five core-backed solvers. No browser UI change, live backend restart, commit, push or release for this update. Backend reload and F5 are still required after deployment. The earlier unrelated synthetic-text smoke failure remains unresolved and is not counted as a pass.

## Shared samplers and SD1.5 LCM Hires — 2026-09-26

- Local Forge HEAD is the pinned `710f1e25fcac84d880cbccf27d11b2e3276e589e`; normalized sampling.py SHA256 matches the existing source manifest. Installed Comfy implementations were compared, including default noise, predictor access and sampler-specific options. Same names are not a blanket equality claim.
- Pinned-source numerical tests cover new LCM/LMS/Res Multistep on SD1.5/SDXL/Anima, txt2img/scaled img2img/exact img2img and sigma-discard variants, plus DPM++ SDE/3M SDE with CPU/NV Brownian replay. All reference comparisons passed. Two old reconstruction tests expected warnings in titles; they were updated to require the requested plain title and diagnostics retained in properties, then passed with the related regression suite (78 passed).
- `tools/core_smoke.py ... --sampling-only`: 24 actual Comfy ModelPatcher/CFGGuider synthetic CPU cases passed; no global sampler/RNG mutations. The broader smoke's separate synthetic text phase failed its finite/repeat assertion and is not counted as a pass. It was not modified as part of this sampler change.
- Frontend: 38 tests passed. Isolated Chromium against the installed frontend loaded the supplied SD1.5 workflow with staged JS: old long title removed, width 690→300, actual pointer open/close of warnings, unchanged prompt inputs, save and F5/reimport, no page errors. No user workflow or server settings were written.
- Real GPU, one-shot process using the supplied WebP metadata and existing `lucSD15_v10` checkpoint/LCM LoRA: 8 LCM/Karras base steps at 512×768, Lanczos 1.5×, 12 LCM/Karras Hires steps at CFG 4.5 / denoise 0.45, Clip Skip 2 and discard-penultimate retained. Completed in 24.30 seconds after imports; output `sd15-lcm-hires-qa_00001_.png`, 768×1152. Visual inspection found similar subject/style but different clothing/composition; not image parity. Other new samplers have no real-model GPU certification.
- The one-shot harness initially omitted Comfy's inference context and failed at Lanczos tensor→NumPy conversion; fixing the test harness to match Comfy execution resolved it without changing core/image-resize code. It exits after the test; the user's 8188 server was not restarted or replaced.

## Explicit Comfy mode and localized compact Settings — 2026-09-26

- Python regression tests: 87 passed (Comfy mode, reconstruction, Hires and contracts). Frontend: 37 passed. Old inputs remain in the same order; the optional compatibility toggle defaults to ON. OFF bypasses the Forge Spec/binding validator and uses connected runtime objects and current widgets.
- `tools/comfy_mode_smoke.py`: actual installed core CLIPTextEncode/CLIPSetLastLayer/KSampler with synthetic conditioning and analytic ModelPatchers. SD1.5, SDXL and Anima at denoise 1.0 and 0.5 matched direct core calls exactly in all six CPU cases. Original CLIP layer settings were not mutated. This does not certify pretrained image or GPU parity.
- Disposable ComfyUI CPU server on 127.0.0.1:8197, isolated user/database/output directories, only Bridge enabled: real frontend pointer toggle, connected Hires synchronization, Japanese/English labels, 300px width, editing, old workflow import, serialization and F5/reimport passed with no page errors. Execution inputs remained identical across reload; core node titles changed normally with Comfy locale.
- User-managed 8188 server was not restarted. Backend updates require its normal restart, followed by browser F5. No new real-model GPU generation or pending Hires GPU rerun was performed for this change.

## Seed controls — 2026-09-26

- Node frontend: 35 passed. New checks cover exact uint64 increments/decrements and bounds, randomization, numeric editing, the old six-value workflow layout, saved control modes, linked/partial execution, and before/after queue timing.
- Real ComfyUI frontend tested with headless Chromium: pointer clicks on both arrows, direct entry of `18446744073709551615`, three queued increments (`9007199254740993` through `9007199254740995`), randomization, workflow serialization and F5/reimport. All sampler inputs survived round-trip unchanged; no page errors.
- `/prompt` submissions were intercepted in the test browser, so the queue lifecycle was exercised without running GPU jobs or writing settings/workflows to the user's server. Backend sampling code is unchanged by this UI update. This does not complete the pending Hires GPU rerun.

## Settings accordions — 2026-09-26

- Node frontend: 28 passed, including original positional workflow round-trips, saved section states, Clip Skip/ENSD visibility, conditional SDXL/img2img sections, unchanged source data, and standalone latent connections.
- Installed ComfyUI frontend tested in disposable headless Chromium against `127.0.0.9:8188`, using the deployed extension files. Real pointer expansion, node height changes (298 to 418 px for the noise section), workflow serialization/reload, full browser refresh followed by workflow reimport, and source information expansion passed with no page errors.
- API prompt inputs matched all imported values before/after folding and reload; only the deliberately edited variation strength changed. No GPU jobs or server-side workflow/settings writes were issued by this UI test. The user's existing browser tab was not reloaded.
- This validates the Settings accordion UI. It does not replace the separate image-drop and GPU parity checks below.

## Compatibility review — 2026-09-26

- Python: 158 passed, 196 optional tests skipped. Node frontend: 22 passed. Skips are not successes.
- Supplied Z-Image and ERNIE WebP files reproduced the old Clip Skip blocker. Forge no-op/pooled-only behavior confirms the fix.
- Qwen text-only / original Edit / 2509 / 2511 reference routes, Flux.1 distilled guidance, missing assets, aliases, and discard-sigma step counts have regression tests.
- Text-only Qwen sampling was compared to installed Forge and Comfy sources. Official edit blueprints are not equivalent to text-only Forge generation.
- Flux.1 has no installed model for GPU validation. Original Qwen Edit/2509 reference runs and pixel parity are unverified. Native workflows retain source metadata but do not implement every optional Forge control.
- Live reconstruct endpoint returned HTTP 200 for all three supplied WebP files after the managed ComfyUI restart. Z-Image retains missing `qwen3_4b_f32-q8_0`; ERNIE retains missing `ernie-image-turbo` rather than substituting installed derivatives.
- Qwen 2511 int8 + Lightning: real GPU success, prompt `e59550a8-b84a-4ff6-a875-ff4eb66e31aa`, 1024x1024, 8 LCM steps, seed 706933366, 55.89 seconds. Output: `ForgeNeo-Bridge/qwen2511-compat_00001_.png`. Composition is similar, not pixel-identical.
- User-approved Z-Image substitute: `qwen_3_4b.safetensors` instead of the missing Q8 encoder. GPU and visual check passed at 1024x1344, Euler/simple 9 steps, seed 1821428978 (45.87 seconds, prompt `ec524d3c-8d74-4806-a186-e386da464229`). Output: `ForgeNeo-Bridge/zimage-compatible_00001_.png`.
- User-approved ERNIE substitute: `ernie-image-turbo-QT-v2-mxfp8.safetensors`. Initial execution completed but produced a repeating texture: AuraFlow had overwritten ERNIE's timestep multiplier with 1 instead of 1000. Changing only that patch to ModelSamplingSD3 restored a recognizable cyclist/coast image at 848x1264, Euler/simple 8 steps, seed 2223788929 (7.62 seconds with models cached, prompt `5c9bf862-44ba-467c-88b0-07e14a72bf47`). Output: `ForgeNeo-Bridge/ernie-compatible-sd3_00001_.png`. This is a successful compatible-model run, not original-weight parity.
- Live canvas/drop/F5 verification remains blocked: the Chrome test tab returned `ERR_BLOCKED_BY_CLIENT`. Backend execution and mocked frontend tests do not establish live browser success.

## 2026-09-26 local image-to-workflow change

The new drop path was implemented and tested in a local work copy. Python pytest:
124 passed, 196 skipped (reference/oracle source unavailable). Node.js: 19 passed.
`node --check web/bridge.js` and `git diff --check` passed. Tests cover missing
models, exact asset matching, edit-time Loader/LoRA values, sampling scope,
unsupported metadata, direct drop, concurrent changes and rollback.

The Chromium harness was updated for direct drop, but could not run here because
Playwright is not installed in the local test Python. Live Comfy registration,
browser save/reload, pretrained GPU generation, and image parity are still
unverified. The earlier numbers below describe the historical 1.0.0 validation
and were not rerun as part of this local change.

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
