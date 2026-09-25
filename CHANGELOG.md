# Changelog

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
