# ComfyUI-ForgeNeo-Bridge 技術資料

[画像付きの使い方・English](../README.md)へ戻る。この資料は詳細なサンプラー比較・互換処理・検証方法をまとめたものです。

**Forge画像をcanvasへドロップすると、編集可能なComfyUIワークフローを組み立てる互換ブリッジです。**
Animaを優先し、SD1.5・SDXL Baseのテキスト処理、乱数系列、Sampler/Sigma/CFG、画像メタデータの読込みを扱います。

**実装版 1.0.0。実モデルGPUのStrong Parity認証は未実施です。**
CPU参照比較、実Comfyクラスの小型モデルAPI試験、ブラウザーの独立UI試験と、実画像生成の一致保証を区別しています。
H3/LTXの実生成による無干渉Gateも未実施です。認証済みモデル一覧はまだ空です。

- 互換動作の正本: Forge Neo [`710f1e25`](https://github.com/Haoming02/sd-webui-forge-classic/commit/710f1e25fcac84d880cbccf27d11b2e3276e589e)
- 読み取った実装仕様: [SPEC_JA.md](SPEC_JA.md)
- 確認結果と残る検証: [VALIDATION.md](VALIDATION.md)
- ライセンス: AGPL-3.0。移植・同梱ソースは [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) を参照してください。

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

### 1. Forge画像をドロップ

ComfyUIのcanvasへForge Neo/A1111のPNG、WebP、JPEGをドラッグしてください。画像に記録されたPrompt、Seed、Steps、CFG、Sampler、サイズなどを読み、モデル系統に応じて標準LoaderとLatentを配線したグラフを直接配置します。専用ダイアログや事前のモデル選択はありません。生成は通常のQueue操作で開始します。標準ノードの表示名はComfyUIの元の名前を保ちます。

モデル、Text Encoder、VAE、LoRAはComfyUIが登録したフォルダから、相対パス、完全ファイル名、拡張子を省いた完全stemの順に照合します。候補が一意でない場合やファイルがない場合は、元の名前をLoaderへ残して不足を表示します。先頭候補を勝手に採用せず、ドロップ時に重みをGPUへロードしません。足りないファイルはLoaderを編集してからQueueしてください。

設定 **`ForgeNeo Bridge: canvas image to workflow`** は初期ONです。新しい `ForgeNeo.Bridge.enable_canvas_drop` への移行時も初回ONになり、その後に変更したON/OFFは保持します。更新後はComfyUIを再起動してブラウザーを再読み込みしてください。Comfyのnative workflow/API promptが埋め込まれた画像と、JSONなど他のファイルは元のハンドラーへ渡します。通常のComfy `IMAGE` TensorにはEXIFが残らないため、元画像ファイルをドロップしてください。

`ForgeNeo Bridge Settings`はForge Neo互換・サンプリング差分補正・Clip Skip・ENSDを常時表示します。モデル種別・生成モードと既存の詳細項目は「詳細設定 / Advanced settings」に収納し、初期状態は閉じます。初期幅は300pxです。SDXL・img2imgの項目は該当時だけ表示します。親・子の開閉状態をワークフローに保存し、閉じても値は保持します。以前に詳細を開いて保存したファイルでは親も開いて復元します。読み込み警告と「読み込み情報 / Source information」は親の外から独立して開閉できます。接続されたLatentがある場合、サイズはLatent側で操作します。Latent未接続の旧ワークフローではSettingsのサイズ欄を残します。標準Loader・Samplerの名前と通常の操作項目は維持します。Bridgeの設定ラベル・アコーデオン・編集画面はComfyUIの言語設定に従い、日本語または英語で表示します。

**配置の整理:** canvasまたはノードを右クリックし、**ForgeNeo Bridge → 配置を整える / Arrange workflow**を選びます。常時ON/OFFする設定ではなく、その場で一度だけ配置を整える操作です。モデル・サイズ、プロンプト、基本生成、Hires、出力を控えめなグループ枠でまとめ、Settingsを対応するSamplerの下へ配置します。HiresのDecode→拡大→Encodeは接続順で並べます。名前・設定値・接続は変えず、Ctrl+Zで配置・枠・表示サイズをまとめて戻せます。詳細を開いて他のノードを自動で動かすことはありません。

Samplerのプレビューが縦に伸びることを見込み、展開中のSamplerには**高さ500px以上の領域＋下側60pxの間隔**を確保します。既に500pxより高ければ実際の高さを使います。Save Image / Preview Imageは**幅510px・高さ650px以上**に広げ、それ以上に手動で広げたサイズは維持します。その他のノードサイズと折りたたみ状態は維持します。グループ枠は上100px・左右と下30pxの余白を取り、グループ名がノード上部の表示と重なりにくくします。

既知の制約：通常の再整理では枠を置き換えますが、グループごとのコピー／貼り付け後は古いグループIDが残るため、枠の重複や別の手動枠の削除が起こる問題が未修正です。コピー後の整理は避けてください。

複数のBridgeワークフローがある場合、対象のBridgeノードを1つ選択するか、そのノードを右クリックしてください。複数ノードを明示選択した場合はその選択範囲を整理します。Loaderを共有している旧ワークフローなど対象が曖昧な場合は、対象ノード一式の選択を案内します。無関係なノードと手作業で作ったグループは動かしません。

Forge画像から**新規に再構築**する際には、この配置を自動適用します。ComfyUIワークフロー付き画像や保存済みJSONは保存された位置を復元し、自動では並べ替えません。必要なときに右クリックの操作を使ってください。

**Forge Neo互換は初期ON、サンプリング差分補正は初期OFFです。** 互換ONではForge用テキスト・初期ノイズを使い、サンプリングの内部処理は独立した補正スイッチで選びます。互換OFFでは同じBridgeノードの内部でComfyUIの`CLIPTextEncode`、明示したClip Skipの`CLIPSetLastLayer`、`KSampler`を呼び出します。現在のPrompt・Seed・Steps・CFG・Sampler・Scheduler・denoiseと、接続したモデル／Latentを使います。元画像の設定は出典として保持し、編集値へ戻しません。OFFでもBridgeのインストールは必要です。標準ノードへの変換機能ではありません。

Bridgeの全8ノードの各入力・出力・説明、およびSettingsのアコーデオンとシード更新コントロールにマウスオーバーヘルプがあります。ComfyUIの言語設定が日本語なら日本語、それ以外は英語で表示します。ComfyUIの「Enable Tooltips」がONのときに表示されます。

OFFではENSD、Forge RNG、強調方式、詳細Sigma/CFG、SDXL追加条件などのForge専用項目を保存したまま「適用なし」と表示します。Clip Skipの効果は読み込んだComfyエンコーダーの対応に従います。Schedulerの`automatic`/`uniform`はComfyの`normal`、`ddim`は`ddim_uniform`として扱います。これらは同一性を保証する変換ではありません。保存済みの`unipc`はComfyの`uni_pc`に対応します。選択肢にはインストールされたComfyUIの一覧も含みます。ONでは下記17種を使用でき、OFFではComfyUIが実装する項目を使用できます。どちらにも実行経路のない項目は具体名付きのエラーとし、別サンプラーへ変更しません。両モードの画素一致は保証しません。

Hires.fixの前後段では、Settingsの互換または差分補正スイッチを画面上で変更すると、Latent／画像経路でつながった相手の同じスイッチも連動します。サイズ変更と2段目の処理は残り、補正OFFではdenoiseとステップ数を標準KSamplerの解釈で処理します。別ワークフローへの変換や再配線は行いません。

ERNIE、Z-Image、Qwen-Image（2.1を含む）、Krea2、Flux.1、Flux.2 Klein/Devでは、主にComfyUIの標準ノードでモデル別グラフを生成します。ERNIEのText Encoderは`flux2`、Z-Imageは`lumina2`、Qwenは`qwen_image`、Krea2は`krea2`、Flux.2は`flux2`のLoader形式です。ERNIE/Flux.2は`EmptyFlux2LatentImage`、Z-Image/Qwen-Image 1.x/Flux.1は`EmptySD3LatentImage`、Krea2/Qwen-Image 2.1は`EmptyLatentImage`を接続します。Flux.1のCLIP-L/T5には`DualCLIPLoader`を使用します。Flux.2は標準`Flux2Scheduler`と`SamplerCustomAdvanced`の経路です。

この標準ノード経路は**編集・実行可能なComfyUIワークフロー**の生成であり、Forge Neoの乱数や全詳細設定のピクセル一致を保証しません。見つからないモデルは別系統のCheckpointで代用せず、元の名前を`UNETLoader`へ残します。旧版で生成済みの`ForgeNeoBridgeKSampler`を含むワークフローは自動変換されないため、元画像を再ドロップしてください。動画Wan、および現環境で検証できないLumina2/PiDは誤ったグラフを作らず、レシピ未対応のエラーを返します。

### モデル別の互換性と未検証範囲

| 系統 | 復元する処理 | 検証範囲 |
|---|---|---|
| Z-Image / ERNIE | 非適用のClip Skipで停止しない。Z-ImageはModelSamplingAuraFlow、ERNIEはModelSamplingSD3でShiftと時刻尺度を保持 | ユーザー承認の互換モデルでGPU生成・出力目視確認済み。Z-Image：Qwen3 4B safetensors、1024×1344、9ステップ。ERNIE：QT v2 MXFP8、848×1264、8ステップ。元の重みとの完全一致は未検証 |
| Qwen-Image / Edit 2511・2509 | 参照なしはCLIPTextEncode、編集時はTextEncodeQwenImageEditPlus。モデル内蔵Shiftを保持 | 2511 int8 + Lightning LoRAの1024×1024、LCM 8ステップがGPU完走（55.89秒）。2509と編集用参照画像経路はGPU未検証。画像の完全一致は未達 |
| Qwen-Image-Edit 初代 | 編集時はTextEncodeQwenImageEdit | 仕様・配線テストのみ。GPU未検証 |
| Flux.1 Dev / Schnell | DualCLIPLoader、DevのDistilled CFG、Schnellの4ステップ初期値 | モデル未所持。公式仕様・配線テストのみ、GPU未検証 |
| Qwen-Image 2.1 / Krea2 / Flux.2 | 既存のモデル別レシピ | 今回のGPU再検証は未実施。Qwen 2.1の参照画像編集は未対応 |

Clip SkipはLLM系ではForge側も非適用です。Flux.1も使用するCLIP pooled出力に影響しないため、T5へ層スキップを適用しません。値を出典へ残し、読み込みを継続します。通常CFGとFlux.1のDistilled CFGは別々に反映します。

`Discard penultimate sigma: True` のある標準サンプラー経路では、`ForgeNeoBridgeScheduler` でsigma列の末尾直前を除去し、標準 `SamplerCustomAdvanced` へ接続します。指定がない通常経路は標準KSamplerです。Flux.2専用Schedulerでの除去指定は未再現です。

生成済み画像から編集元画像は復元できません。img2imgではLoadImageで元画像を指定します。元のInfotextと注記は `extra.forge_neo_bridge` に保存します。GGUFなど追加Loaderが必要な形式の標準Loaderでの実行、強調構文・追加ノイズ・詳細schedulerオプションの完全再現は保証しません。

照合資料: [ComfyUI Flux.1](https://docs.comfy.org/tutorials/flux/flux-1-text-to-image)、[ComfyUI Qwen 2511](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)、[Qwen Edit 2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)、ローカルForge Neoのdiffusion_engine / text_processing実装。

### 2. 生成されたノードを編集

| class ID | 表示／役割 | 主な出力 |
|---|---|---|
| `ForgeNeoBridgeTextEncode` | ForgeNeo Bridge Text Encode — 正負Promptを各ノードで編集 | `FORGE_TEXT_INPUT` |
| `ForgeNeoBridgeSettings` | ForgeNeo Bridge Settings — RNG、ENSD、Clip Skip、詳細値 | `FORGE_SETTINGS` |
| `ForgeNeoBridgeKSampler` | ForgeNeo Bridge KSampler — Seed、Steps、CFG、Sampler、denoise | `LATENT` |

txt2imgでは標準の`EmptyLatentImage`がKSamplerの`input_latent`へ接続され、サイズとバッチ数はこのLatentノードで編集します。img2imgでは`LoadImage → VAEEncode`のLatentを接続します。Settingsに残るサイズ・バッチ数は、接続のない旧ワークフロー用の詳細設定です。

SD1.5・SDXL・AnimaのHires.fixは、対応する設定を2段階のワークフローへ復元します。Lanczosは`1回目のSampler → VAEDecode → ImageScale → VAEEncode → Hires Sampler → VAEDecode`です。1回目はdenoise=1、2回目は記録されたHires steps・CFG・denoiseを使います。Sampling adjustmentsがONならForgeのexact_steps方式、OFF（初期値）ならComfyUI標準のステップ・denoise解釈です。Seed・RNG・ENSD・Clip Skip・Eta・sigma除去設定を両方に保持し、適用の有無は互換・補正スイッチに従います。SDXLの2回目のサイズ条件は接続Latentに追従します。AnimaのHires Shiftは2回目のSettingsへ反映し、補正ONで適用します。Hires用Prompt・Sampler・Schedulerの変更と、同じCheckpoint/Moduleを使う指定にも対応します。

Latent拡大は`Latent`（bilinear）、`Latent (bicubic)`、`Latent (nearest-exact)`に対応し、標準`LatentUpscale`を接続します。別Checkpoint/Module/LoRAへの切替、学習型upscaler、antialias付きLatentは未対応です。BridgeのHiresレシピが扱えない設定では、理由付きのエラーを返し、通常のtxt2imgに見える不完全なグラフを作りません。

最終サイズは明示された`Hires resize`を優先します。倍率指定の元画像をドロップした場合は実画像の幅・高さを使用し、単純な倍率計算と異なる場合は読み込み情報へ記録します。画像が後からリサイズされている場合も、その実寸が採用されるため原本を使用してください。画像寸法を伴わないテキストからの構築では倍率計算を推定値として記録し、Forgeの丸め結果とは断定しません。拡大先は8の倍数である必要があります。**旧版で作ったHiresワークフローは、更新・再起動・F5後に元画像を再ドロップしてください。** SD1.5のLCM＋Lanczos HiresはGPU完走を確認済みです。SDXL HiresとLatent拡大はGPU未検証です。Forge出力とのピクセル一致は保証しません。

Animaの`waiANIMA_v10Base10`＋Lanczos Hires（1024×1344→1536×2048、30+20ステップ）は、Sampling adjustmentsのOFF／ONそれぞれで実モデルGPU生成完走を確認しました。元Forge画像との構図・細部の差は残ります。

MODEL/CLIP/VAE/LoRAは標準ComfyUI Loaderです。Loader名やLoRA強度を後で変更すると、**その時点の接続値**が次の実行に使われます。元画像のデータはSettingsに出典として保存され、実行設定を上書きしません。LoRAを追加するときはPromptタグではなくLoaderを配線します。

Bridge KSamplerのSeedは、標準KSamplerと同じ左右の矢印・ドラッグ・クリック入力で編集できます。直下の`control after generate`で`fixed / increment / decrement / randomize`を選択します。画像の読込直後と旧ワークフローの初期値は`fixed`です。ComfyUI側の生成前／生成後の更新設定に従い、連続キューでも更新します。Seedは内部で64bit整数の文字列を保持するため、大きな値も桁落ちしません。制御モードはワークフローに保存され、Steps・CFGなど既存の保存順は変えません。画面更新はF5で反映されます。

`Module 1/2`の番号だけでTE/VAEを決めません。未知のSamplerや未対応のHiresなどは情報を保持し、対応する値へ修正するまで実行時にエラーを出します。未記録項目にはモデル別テンプレートの初期値を使い、元画像の実値と区別します。

### 3. 保存済みの旧4ノード

従来の`ForgeCompatSpec`、`ForgeCompatText`、`ForgeCompatNoise`、`ForgeCompatSampler`と旧APIは互換のため残しています。旧Specの編集画面はSpecノードの右クリックから開けます。

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
| Sampler | Euler、Euler a、ER-SDE、DPM++ 2M、DPM++ 2M SDE、DPM2、Heun、LCM、LMS、DPM++ SDE、DPM++ 3M SDE、Res Multistep |
| 同名サンプラーの追加経路 | DDIM、UniPC、Euler CFG++、Euler a CFG++、DPM++ 2M CFG++。ComfyUIのsolverを利用。Forgeとの数値一致は未認証 |
| Scheduler | Automatic、Karras、Exponential、Polyexponential、Normal、Simple、Uniform、SGM Uniform、Linear Quadratic、KL Optimal、DDIM、Align Your Steps、Beta、Turbo、Bong Tangent、FlowMatchEulerDiscrete（既定オプション） |
| Noise | CPU／GPU（CUDA）／NV、ENSD、variation、batch seed列、RNG snapshotの分岐・再実行 |
| Sampling | Eta、Churn、tmin/tmax、s_noise、sigma範囲/rho、Beta α/β、discard、SGM、Shift、NGMS、Skip Early CFG |
| Text | モデル別Clip Skip、Emphasis、AND、BREAK、prompt editing、SDXL pooled／size条件／空Negative |
| Metadata | PNG text、JPEG/WebP EXIF UserComment、Unicode、native優先、サイズ・再帰・パス制限 |

上の一覧は**実装経路**であり、全モデル・全組合せのGPU再現性認証ではありません。例えばflowで不正なsigma列ができる組合せは、勝手に修正せず拒否します。

21 sampler名／17 scheduler名をInfotextの識別用に登録していますが、実行可能なのは上記の候補経路です。
上記の復元レシピ以外のHires fix、Inpaint mask、Refiner、Textual Inversion、Anima参照画像、動的LoRA、SDXL RF/v-pred派生、および上記の新しいモデル系統でのForge Neoサンプリング再現は、現時点では未対応表示・原情報保存までです。
それらを黙って外して別の生成に変換しません。

### ComfyUIと共通するサンプラー／スケジューラー

**同名でも内部処理が異なることだけを理由に、使用を禁止しません。対応関係がある項目を使えるようにし、差分と検証範囲をここに記載します。** 未確認の項目を「別物」と断定することもありません。

比較対象はForge Neo `710f1e25`とComfyUI `830232b8`です。更新後は実装が変わる可能性があります。「ほぼ互換」は基本式・通常設定の対応を指し、全モデル・全設定・画素の一致認証ではありません。サンプラーは各ステップの更新方法、スケジューラーは各ステップで使うノイズ量（sigma）の並びです。同じシードでも初期ノイズ、途中の乱数、Prompt処理、モデル／VAEが違えば画像は変わります。

#### Sampling adjustments（サンプリング差分補正）

Sampling adjustmentsは、Forge NeoとComfyUIの**サンプリング処理の互換性を高めるための補助設定**です。同名のサンプラーやスケジューラーでも内部の計算、乱数、ノイズ量の並びなどが異なるため、共通部分を利用しつつ、再現性に影響する処理差をBridgeで補います。

**すでに処理が一致し、互換性が取れている共通部分には追加補正は不要です。それらを使用するためにSampling adjustmentsをONにする必要はありません。** 補正が必要なのは、再現性に影響する処理差がある部分です。ただし、サンプラーの基本式が共通でも、途中の乱数やスケジューラーなどに差が残る場合があるため、サンプラー名だけで「ON/OFFしても結果が変わらない」とは判断できません。

**互換性の向上は、生成画像の一致や、近似した・見た目の近いレンダリング結果を保証するものではありません。** ONにすれば必ずForgeの出力へ近づく、という意味でもありません。同じモデル・プロンプト・シード・ステップ数でも、数値精度、実行環境、エンコーダーやVAE、未対応の処理差などによって、構図・色・細部が変わる場合があります。目的は処理と設定の対応を改善することであり、画像の類似度や品質の保証ではありません。

Settingsの`Forge compatibility`の直下に配置した独立スイッチです。新規ノード・画像ドロップは**初期OFF**です。この項目がない旧ワークフローは、従来の計算を保つためONで読み込みます。保存後のON/OFFは維持します。旧4ノードのSpecも、明示指定がなければ従来の補正経路を使います。

| Forge compatibility | Sampling adjustments | 実行する処理 |
|---|---|---|
| OFF | 適用なし | ComfyUI標準のテキスト処理・ノイズ・KSampler |
| ON | OFF（初期値） | Forgeのテキスト処理・Clip Skip・初期ノイズを保ち、ComfyUIのサンプラーとsigma列を使用 |
| ON | ON | 上記にForge向けのサンプリング・sigma・途中乱数・CFG補助の補正を適用 |

補正OFFではENSD、Eta、Churn、sigma範囲／rho／Betaパラメーター、手動discard、SGM倍率、Shift、NGMS、Skip Early CFG、Forgeのimg2imgステップ方式と追加ノイズは**保存のみで非適用**です。Clip Skip、乱数方式・サブシード・シード基準サイズ、テキスト処理、SDXL条件は互換ONなら残ります。実行レポートに使用経路と非適用項目を記録します。ComfyUIのAutomatic／Uniformは`normal`への対応付け、DDIMは`ddim_uniform`への名称対応を使います。Forge専用スケジューラーを使う場合は補正ONにしてください。別の方式へ黙って置換はしません。

English: **Sampling adjustments is OFF by default.** It improves internal sampling compatibility; it does not guarantee identical or visually similar images, and enabling it need not bring the result closer to Forge. Forge compatibility remains a separate master switch. Older workflows without this field load with adjustments ON to preserve existing behavior.

Shared processing that is already compatible needs no additional correction and does not require Sampling adjustments to be ON. Corrections address differences that affect reproducibility. A shared sampler formula alone does not imply identical intermediate noise or schedules, so the sampler name does not guarantee unchanged results when toggling this option.

#### サンプラーの対応一覧

以下は**互換ONかつ差分補正ON**の一覧です。「Forge経路」は固定版Forgeを移植した12種、「Comfy併用」はComfyUIのsolverを呼ぶ5種です。差分補正OFFでは、この17種もComfyUIのsolver・既定オプションを使います。

| 分類 | Forge表示名 → ComfyUI名 | 互換ONの経路と注意事項 |
|---|---|---|
| 基本式はほぼ共通 | Euler → `euler` | Forge経路。Churnを0より大きくすると途中の乱数処理も比較が必要 |
| 基本式はほぼ共通 | Heun → `heun` | Forge経路。追加ノイズ・Churnの設定を保持 |
| 基本式はほぼ共通 | LMS → `lms` | Forge経路。sigma列をそろえることが前提 |
| 基本式はほぼ共通／表記違い | DPM2 → `dpm_2` | Forge経路。AutomaticはKarras、Forge既定の末尾直前sigma除去あり |
| 基本式はほぼ共通／表記違い | DPM++ 2M → `dpmpp_2m` | Forge経路。AutomaticはKarras |
| 基本式はほぼ共通／表記違い | Res Multistep → `res_multistep` | Forge経路。通常の非ancestral・非CFG++版。派生版とは区別 |
| 更新式は共通、乱数処理に差 | Euler a → `euler_ancestral` | Forge経路。private RNG、ENSD、Eta、s_noiseを保持 |
| 更新式は共通、乱数処理に差 | LCM → `lcm` | Forge経路。途中の再ノイズもprivate RNGから生成 |
| 共通方式、乱数・モデル依存処理に注意 | ER SDE → `er_sde` | Forge経路。Comfy版へ単純に置き換えて乱数一致とは扱わない |
| 共通方式、Brownian noise等に注意 | DPM++ SDE → `dpmpp_sde` | Forge経路。2回評価、AutomaticはKarras、private Brownian noise |
| 共通方式、Brownian noise等に注意 | DPM++ 2M SDE → `dpmpp_2m_sde` | Forge経路。AutomaticはExponential。`_heun`／`_gpu`派生は別選択 |
| 共通方式、Brownian noise等に注意 | DPM++ 3M SDE → `dpmpp_3m_sde` | Forge経路。AutomaticはExponential、既定の末尾直前sigma除去あり |
| 同名だが実装経路が異なる | DDIM → `ddim` | Comfy併用。ComfyのDDIM選択はEuler solverを使用。Forge専用DDIMの時刻処理・Eta DDIMを再現しない。Eta DDIMとChurnは保存するが非適用 |
| 表記違い、終端処理等に差 | UniPC → `uni_pc` | Comfy併用。旧保存名`unipc`も受付。bh1を使用し、Forge既定の末尾直前sigma除去を保持。Forgeは終端が0.001未満なら補正、確認したComfy版は0の場合だけ補正。solver全体のForge数値一致は未検証 |
| 同名方式、完全な同等性は未検証 | Euler CFG++ → `euler_cfg_pp` | Comfy併用。ComfyのCFG++処理を使用。通常のEulerへ置換しない |
| 同名方式、完全な同等性は未検証 | Euler a CFG++ → `euler_ancestral_cfg_pp` | Comfy併用。negative予測をCFG++へ渡し、途中ノイズはprivate RNG＋ENSDを使用 |
| 同名方式、完全な同等性は未検証 | DPM++ 2M CFG++ → `dpmpp_2m_cfg_pp` | Comfy併用。通常のDPM++ 2Mへ置換しない。ForgeのCFG++登録処理に従い、AutomaticはKarras |

Comfy併用でも、Forgeテキスト処理・適用対象モデルのClip Skip・初期乱数・sigma列・Hiresのexact_stepsは残します。途中ノイズが必要なEuler a CFG++にはprivate RNGを渡します。DDIM・UniPC・非ancestralのCFG++では、ENSDを保存していても途中ノイズを追加しないため出力に作用しません。Forge専用DDIMのEtaはComfy solverに適用されません。実行レポートの`sampler_implementation`と`not_applied`に使用経路・非適用設定を記録します。**Comfy併用は、Forgeとの完全一致も、標準KSamplerを単独で使った場合との完全一致も保証しません。**

#### スケジューラーの対応一覧

この表のON/OFFは、互換ONのときの**Sampling adjustments**を指します。互換OFFもComfyUI側の経路です。

| 分類 | Forge表示名 → ComfyUIの選択／ノード | 注意事項 |
|---|---|---|
| 基本式はほぼ共通 | Karras → `karras` | sigma_min/max、rho（通常7）、ステップ数をそろえる |
| 基本式はほぼ共通 | Exponential → `exponential` | sigma範囲・ステップ数をそろえる |
| 基本式はほぼ共通 | Simple → `simple` | モデル側のsigma表が同じことが前提 |
| 基本式はほぼ共通／表記違い | SGM Uniform → `sgm_uniform` | 同じモデルの時間・sigma変換が前提 |
| 基本式はほぼ共通 | Beta → `beta` | α/βとモデル側sigma表をそろえる。重複除去で実際の区間数が減る場合がある |
| 基本式はほぼ共通／表記違い | Linear Quadratic → `linear_quadratic` | 既定の分割数・threshold・sigma_maxが同じ場合 |
| 同名だが条件により差 | Normal → `normal` | Comfyは終端sigmaがほぼゼロのモデルでステップ配置を調整。Forgeには同じ分岐がない |
| 表記違い、条件により差 | DDIM → `ddim_uniform` | Comfy側にほぼゼロのsigmaを考慮する分岐がある。ONではどちらの表記もForgeの`ddim`計算を使用。DDIMサンプラーとは別設定 |
| 同名だがsigma列が異なる | KL Optimal → `kl_optimal` | Forgeはn+1点でsigma_minまで計算。Comfyはn点の後にゼロを追加。終端と途中の間隔が異なる。ONではForgeの列を維持 |
| 独立した同名選択なし | Uniform → OFF時は`normal` | 同一処理とは保証しない。ONではForgeのUniformを使用 |
| 自動選択規則の違い | Automatic → OFF時は`normal` | ONではサンプラー・モデルごとのForge既定を使用。Automatic自体が共通の数式を表すわけではない |
| 専用ノードで対応 | Polyexponential → `PolyexponentialScheduler` | 通常KSamplerの選択肢にはない。ONのBridgeはForge経路で対応。OFFでは自動配線しない |
| 専用ノードがあるが列に差 | Align Your Steps → `AlignYourStepsScheduler` | Forgeのモデル最大sigmaから作る値と、Comfyの固定表・補間／終端処理は同一ではない。ONはForge経路。OFFでは自動配線しない |
| 専用ノード、同等性未検証 | Turbo → `SDTurboScheduler` | 通常KSamplerの選択肢にはない。ONはForge経路。OFFでは自動配線しない |
| 同名の通常選択なし | Bong Tangent／FlowMatchEulerDiscrete | ONはForge経路（対応済みオプションのみ）。OFFで近い名前のschedulerへ置換しない |
| モデル専用の対応 | Flux2 → `Flux2Scheduler` | Flux.2の標準ノード復元レシピで使用。SD1.5/SDXL/Anima用Bridgeのscheduler選択とは別 |

#### 近い名前だけで対応付けない項目

`DPM++ 2s a RF`／`Flux Realistic`を通常の`dpmpp_2s_ancestral`へ、PLMSをLMSへ、RestartやKohaku LoNyu Yogを別samplerへ自動変換しません。この4系統には今回のBridge実行経路がありません。`uni_pc_bh2`、`_gpu`／`_heun`付きSDE、その他Comfy固有の選択は互換OFFで利用できますが、Forgeの同名処理とは扱いません。未対応は「数学的に別物と証明済み」という意味ではなく、必要な対応経路がまだないという意味です。

実装の参照先: [Forge sampler登録](https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/modules/sd_samplers_kdiffusion.py)、[Forge scheduler](https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/modules/sd_schedulers.py)、[Comfy sampler/scheduler](https://github.com/Comfy-Org/ComfyUI/blob/830232b856045ca2892833212d7771078a13edd5/comfy/samplers.py)、[Comfy solver](https://github.com/Comfy-Org/ComfyUI/blob/830232b856045ca2892833212d7771078a13edd5/comfy/k_diffusion/sampling.py)。

LCMのSD1.5＋LCM LoRA＋Lanczos Hires（512×768→768×1152、8+12ステップ）は実モデルで生成完走を確認しました。元Forge画像とは構図・衣装に差があり、画素一致ではありません。同時に追加したForge経路のLMS・DPM++ SDE・DPM++ 3M SDE・Res MultistepはForge参照のCPU数値比較と実Comfyクラスによる合成モデル試験までで、実モデルGPU再現性は未検証です。

Comfy併用5種はSD1.5・SDXL・Animaの合成モデル、CFG 1/2、txt2img／exact_steps img2imgの60ケースでCPU実行・有限値・再実行一致・共有状態の保持を確認しました。これら5種のForge参照との数値一致、実モデルGPU生成、画像品質は未検証です。

Settingsのタイトルには警告や推定情報を付けません。旧版の自動付加タイトルも読み込み時に`ForgeNeo Bridge Settings`へ戻し、幅を300pxへリセットします。警告はImport warningsから開閉でき、同じボタンの再クリックで閉じます。

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

[検証手順と結果](VALIDATION.md)、[未実施の正式テストマトリクス](test_matrix.json)、[例Spec](../examples)を参照してください。
