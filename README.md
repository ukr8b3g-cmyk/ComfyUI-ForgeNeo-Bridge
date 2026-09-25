# ComfyUI-ForgeNeo-Bridge

**Forge Neoの生成設定を、ComfyUIの専用4ノードへ渡す互換ブリッジです。**
Animaを優先し、SD1.5・SDXL Baseのテキスト処理、乱数系列、Sampler/Sigma/CFG、画像メタデータの読込みを扱います。

**実装版 1.0.0。実モデルGPUのStrong Parity認証は未実施です。**
CPU参照比較、実Comfyクラスの小型モデルAPI試験、ブラウザーの独立UI試験と、実画像生成の一致保証を区別しています。
H3/LTXの実生成による無干渉Gateも未実施です。認証済みモデル一覧はまだ空です。

- 互換動作の正本: Forge Neo [`710f1e25`](https://github.com/Haoming02/sd-webui-forge-classic/commit/710f1e25fcac84d880cbccf27d11b2e3276e589e)
- 読み取った実装仕様: [SPEC_JA.md](docs/SPEC_JA.md)
- 確認結果と残る検証: [VALIDATION.md](docs/VALIDATION.md)
- ライセンス: AGPL-3.0。移植・同梱ソースは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

## インストール

ComfyUIの `custom_nodes` 内で実行します。

```powershell
git clone https://github.com/ukr8b3g-cmyk/ComfyUI-ForgeNeo-Bridge.git
```

ComfyUIを再起動し、ブラウザーも再読み込みしてください。更新はリポジトリ内で `git pull --ff-only` です。

**起動時のpip実行、モデルの自動ダウンロード、ComfyのSampler/RNG関数の書換えは行いません。**
通常のComfyUI環境のPyTorch、NumPy、SciPy、Pillow、tqdmを利用します。DPM++ 2M SDE経路はComfyUIの既存依存 `torchsde==0.2.6` を検査します。別バージョンを自動的に置換しません。
Prompt parser用Lark 1.3.1は、MITライセンスのソースをprivate packageとして同梱しています。外部の `lark` packageを置換・更新しません。

smZNodesなど、すでにグローバル関数を変更する拡張が入っている環境では、そちらの影響をこの拡張が取り消すことはありません。最初の比較は、そのような拡張を無効にしてComfyUIを再起動した環境で行ってください。

## 使い方

### 1. Import / Specを開く

メニュー／コマンド **`ForgeNeo Bridge — Import / Spec`** を開きます。
`ForgeCompatSpec` ノードの右クリックメニューからも編集画面を開けます。

1. Model familyをAnima／SD1.5／SDXLから選択します。
2. ForgeのPNG／WebP／JPEG、Infotext、またはSpec JSONを読み込みます。手動作成も可能です。
3. Model／Text Encoder／VAE／LoRAを**正確なファイル名**で選択します。
4. 推定値・未対応項目・不明な設定を確認し、「設定を検証」を実行します。
5. 「新しいワークフローを作成」で標準Loaderと専用4ノードを配置します。画像生成は自動開始しません。

`Module 1/2` の順番や似たファイル名から資産を勝手に決定しません。未解決のままではStrict実行できません。
元画像の設定が欠けている場合も、初期値を「生成に使われた実値」と偽って扱いません。

### 2. Canvasへのドラッグ＆ドロップ

設定 **`ForgeNeo Bridge: opt-in canvas image import`** は初期OFFです。ONにするとForge画像をcanvasへドロップしてレビュー画面を開けます。

Comfyのnative workflow/API promptが埋め込まれた画像はnative読込みを優先します。JSONやその他のファイルは元のハンドラーへ渡します。
Import失敗・キャンセル時に元グラフを先に消す処理はありません。
通常のComfy `IMAGE` TensorにはEXIFが残らないため、メタデータ源には元の画像ファイルを使います。

### 3. 4ノード

| class ID | 表示／役割 | 主な出力 |
|---|---|---|
| `ForgeCompatSpec` | Forge Compat Import / Spec — 設定の正本 | `FORGE_GENERATION_SPEC`、Report |
| `ForgeCompatText` | Forge Compat Text — Text／Emphasis／Schedule | `FORGE_CONDITIONING`、Report |
| `ForgeCompatNoise` | Forge Compat Noise — 初期noiseとprivate RNG snapshot | `LATENT`、`FORGE_NOISE_BUNDLE`、Report |
| `ForgeCompatSampler` | Forge Compat Sampler — Sigma／CFG／Sampling | `LATENT`、`SIGMAS`、Report |

```text
Import / Spec ────────── SpecをText / Noise / Samplerへ
CLIP Loader → LoRA ──── Text ─── Conditioning ───────┐
MODEL Loader → LoRA ── Noise ── LATENT + Noise ─────┤
              └──────────────────── MODEL ─────────┤
                                                  Sampler
                                                     ↓
VAE Loader ───────────────────────────────────── VAE Decode → Save
```

専用Conditioning/Noiseを標準KSamplerへつなぐことはできません。Prompt scheduleや乱数継続が黙って失われるのを防ぐためです。
LoRAは標準の明示Loader経路で1回だけ適用します。Sampler内部の自動LoRA読込みはありません。
Spec、noise、conditioningの設定hashや入力latentが異なる場合は、生成前に停止します。

## V1の実装範囲

| 項目 | 状態 |
|---|---|
| Family | Anima flow／SD1.5 epsilon／SDXL Base epsilon |
| Operation | txt2img、マスクなし基本img2img |
| Sampler | Euler、Euler a、ER-SDE、DPM++ 2M、DPM++ 2M SDE、DPM2、Heun |
| Scheduler | Automatic、Karras、Exponential、Polyexponential、Normal、Simple、Uniform、SGM Uniform、Linear Quadratic、KL Optimal、DDIM、Align Your Steps、Beta、Turbo、Bong Tangent、FlowMatchEulerDiscrete（既定オプション） |
| Noise | CPU／GPU（CUDA）／NV、ENSD、variation、batch seed列、RNG snapshotの分岐・再実行 |
| Sampling | Eta、Churn、tmin/tmax、s_noise、sigma範囲/rho、Beta α/β、discard、SGM、Shift、NGMS、Skip Early CFG |
| Text | モデル別Clip Skip、Emphasis、AND、BREAK、prompt editing、SDXL pooled／size条件／空Negative |
| Metadata | PNG text、JPEG/WebP EXIF UserComment、Unicode、native優先、サイズ・再帰・パス制限 |

上の一覧は**実装経路**であり、全モデル・全組合せのGPU再現性認証ではありません。例えばflowで不正なsigma列ができる組合せは、勝手に修正せず拒否します。

21 sampler名／17 scheduler名をInfotextの識別用に登録していますが、実行可能なのは上記の候補経路です。
Hires fix、Inpaint mask、Refiner、Textual Inversion、Anima参照画像、動的LoRA、SDXL RF/v-pred派生、Flux/Qwen Image/Krea2などは、現時点では未対応表示・原情報保存までです。
それらを黙って外して別の生成に変換しません。

### Clip SkipとSGM

- **Anima:** Forge NeoのAnima経路と同様、Clip Skipは非適用です。Qwenに一律`-2`を設定しません。
- **SD1.5:** 指定層とFinal Layer Normを再現します。
- **SDXL:** 最小2。L／G、pooled出力とサイズ条件を別々に扱います。保存画像にClip Skipがない場合、実値2だったとは断定しません。
- **SGM noise multiplier:** Scheduler名ではなく初期noiseの倍率規則です。Anima flowへepsilon/v系のsqrt式を流用しません。
- **ENSD:** Eta Noise Seed Delta。追加noise側の乱数系列に関わる値で、sigma値ではありません。

## 精度・Report・制約

通常は数値精度を`as_loaded`として、Loaderの実精度を使います。指定精度を適用できない場合、共有MODEL/CLIPへ無断で`.to(dtype)`する代わりに明示エラーにします。
特にAnimaのQwen／LLM adapter精度は実モデルGPU比較で今後認証する必要があります。

各ノードのReportはJSON文字列です。Samplerの`completed`はsamplingの完了を意味し、画像の再現性合格ではありません。比較していないrunの`qualification`は`not_evaluated`です。
`trace_level=tensors`は診断用です。tensor traceはComfy出力配下へ保存され、prompt/conditioningを含み得るため共有前に内容を確認してください。

GPU不足時に解像度・dtype・steps・タイル方式を自動変更しません。OOMはそのまま報告します。
H3／LTX／動画MODEL・NestedTensorは、この専用Noise/Samplerへ誤接続した場合だけ拒否し、通常の動画ワークフローを走査・変更しません。

## 開発者向け検証

```text
python -m pytest -q
node --test tests/frontend.test.mjs
python tools/core_smoke.py /path/to/pinned/ComfyUI
python tools/browser_smoke.py
```

Forge oracleを用いたCPU比較は環境変数`FORGE_REFERENCE`へ固定Forgeソースのパスを指定してください。未指定・未取得時のoracle試験はSKIPであり、PASSではありません。
`core_smoke.py`は実Comfyクラス＋小型の合成重み／解析用denoiserです。実モデルの画質試験ではありません。
`browser_smoke.py`はChromiumと実Bridge UIを使いますが、Comfyホスト部分はモックです。ライブComfy画面試験とは区別します。

[検証手順と結果](docs/VALIDATION.md)、[未実施の正式テストマトリクス](docs/test_matrix.json)、[例Spec](examples)を参照してください。
