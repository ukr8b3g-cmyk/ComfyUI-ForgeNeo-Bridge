# Changelog

## 2026-09-27 — deep-audit fixes

- Keep negative conditioning for the three CFG++ samplers at CFG 1, including empty negatives and transient CFG 1 controls. Align Qwen Edit positive/negative reference conditioning with ComfyUI's native Edit nodes.
- Preserve independently applied and repeated LoRAs in imported Bridge graphs, including same-name loaders on opposite sides of a shared loader. Validate only active batch/variation/ENSD seed offsets before RNG creation.
- Accept Forge's `Shift: 0` model-default sentinel for Z-Image and ERNIE imports while preserving the original request. Replace only arrangement frames whose saved ownership and current members match; retain manual, partial and unmarked legacy frames.
- Make summary tracing lighter and repair the synthetic text smoke fixture. Update documentation and CPU/frontend regressions without changing the default-OFF Sampling adjustments setting.

## 2026-09-26 — image-to-workflow import and Hires.fix

- Publish a Japanese-first / English-second illustrated README, original-resolution comparison images, cropped node guides and localized arrangement videos. Explain defaults, model-specific verification limits and the remaining Hires.fix output differences.
- Reconstruct Anima Hires.fix with a second sampler and its own Hires Shift. Use explicit Hires dimensions first, then original image dimensions for scale-based imports; report unsupported recipes instead of silently constructing one pass. Verify both adjustment modes with the supplied Anima model; the user confirmed operation.

- Reserve at least 500px height per expanded sampler, plus the normal 60px gap below. Expand SaveImage/PreviewImage to at least 510×650px while retaining larger user sizes. Increase group padding to 100px at the top and 30px on the other sides. Preserve settings, links and saved positions until the explicit arrange operation; Undo includes image-node resizing.
- Add localized canvas/node context action ForgeNeo Bridge → Arrange workflow, with native Undo/Redo. Group model/size, prompts, base generation, Hires and output; put Settings beside its own pass and order Hires image steps by connections. Apply on fresh reconstruction only, preserving saved native layouts. Preserve execution inputs, connections and manual node dimensions; isolate independent Bridge workflows and replace only owned frames.
- Collapse family/mode and detailed accordions under Advanced settings. Keep compatibility, sampling adjustments, Clip Skip and ENSD visible at 300px width; source/warnings remain independently accessible. Persist parent/child open states without changing widget serialization.

- Add independent default-OFF Sampling adjustments under the existing Forge compatibility switch. OFF retains Forge text/initial noise with core Comfy samplers and schedules; ON preserves the prior Forge-adjusted path. Migrate older saved workflows to ON, preserve new choices, and synchronize connected Hires passes. No image-similarity guarantee.
- Add Japanese/English native hover help for all eight Bridge nodes, input/output fields, accordion headers and seed controls, following ComfyUI's locale without widening Settings.

- Document shared sampler/scheduler name mappings, conditional differences (including Normal, DDIM and KL Optimal), and unverified equivalence. Allow DDIM, UniPC and three CFG++ solvers through installed Comfy implementations with Forge text/noise/schedules retained; report core-backed execution and unapplied options. Preserve the imported `unipc` alias in both modes.
- Enable Forge-compatible LCM, LMS, DPM++ SDE, DPM++ 3M SDE and Res Multistep from the existing pinned source. Preserve per-run RNG, private Brownian state, sampler defaults, second-order text timing and sigma discard; accept the Comfy scheduler name `ddim_uniform` as Forge `ddim`.
- Remove generated diagnostic text from Settings titles, repair old wide titled nodes on load, and let Import warnings close on a second click. Preserve diagnostics in node properties.
- Add a default-ON Forge compatibility switch to Bridge Settings. OFF explicitly uses core Comfy text encoding and KSampler with current edits; retain inactive Forge values and synchronize connected Hires passes in the UI. Keep old workflows ON and preserve uint64 seeds.
- Follow Comfy UI Japanese/English preference for Bridge controls and editor text; use 300px initial Settings width and retain accordion state and user resizing.
- Give Bridge KSampler the core numeric Seed widget's arrows and click/drag editing, plus fixed/increment/decrement/randomize controls. Preserve exact uint64 seeds, old widget order, saved control modes, and Comfy's before/after queue preference.
- Reconstruct SD1.5/SDXL Hires.fix as two connected passes, with core Lanczos image resize or supported latent resize. Use full denoise for the base pass, exact Hires steps/CFG/denoise for the second, retained RNG/ENSD/Clip Skip, and enlarged SDXL conditioning. Preserve blockers for unsupported model/module/upscaler changes.
- Group Bridge Settings into persistent accordions; keep Clip Skip and ENSD visible, preserve original widget serialization, and retain size controls for legacy graphs without a connected latent. Conditional SDXL/img2img sections and read-only source information are included.
- Use ModelSamplingSD3 for ERNIE shifts to preserve its 1000x timestep scale; verify GPU generation with user-approved local Z-Image and ERNIE substitutes.
- Accept inert Clip Skip metadata; retain source infotext in native workflows.
- Correct Qwen text-only/reference conditioning, built-in shift, Flux.1 distilled guidance, and core sampling aliases.
- Add explicit discard-penultimate sigma scheduling with standard custom sampling.
- Enable reconstruction on first migration to the new setting key; preserve later user choices.

- Forge PNG/WebP/JPEG canvas drop now builds a connected, editable workflow directly.
- Add Text Encode, Settings and KSampler nodes; current loader, LoRA and widget values drive each run.
- Match local models by exact registered path/name/stem; preserve missing and ambiguous names.
- Keep unknown sampler and unsupported Forge effects visible, without silently running altered settings.
- Preserve the four legacy nodes and APIs; no automatic dependency install, model download or GPU queue.
- Selected live-frontend and pretrained GPU execution checks are recorded in docs/VALIDATION.md. General image parity and end-to-end H3/LTX coexistence remain unverified.

## 1.0.0 — implementation candidate

- Add ForgeCompatSpec, ForgeCompatText, ForgeCompatNoise and ForgeCompatSampler.
- Pin compatibility behavior to Forge Neo 710f1e25 and approved Spec 1.0.0.
- Implement immutable configuration, provenance/review, lossless seed strings,
  strict validation, known-loader bindings and input-bundle identity checks.
- Add private CPU/CUDA/NV streams, ENSD/variation/snapshot replay and isolated
  torchsde Brownian intervals, without default RNG or Core monkey patches.
- Add the audited seven-sampler V1 subset, sigma/CFG controls and basic img2img.
- Add Anima/SD1.5/SDXL text adapters with family-specific layer behavior.
- Add bounded PNG/JPEG/WebP metadata reading, native priority, review/editor,
  detached workflow planning and optional canvas drop (off by default).
- Bundle the private MIT Lark parser; no automatic dependency installation.
- Add CPU oracle, contracts, text, metadata, Node.js and independent Chromium
  tests, plus actual pinned Comfy API smoke tests with analytic/tiny models.
- GPU/pretrained image parity, live Comfy frontend and H3/LTX end-to-end gates
  remain untested; this release does not certify Strong Parity.
