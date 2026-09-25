# Forge Compatibility V1 — 確定仕様

仕様版: 1.0.0 / 確定日: 2026-09-25

状態: **設計確定。カスタムノード実装、ローカル反映、GPU A/B、ブラウザー検証、公開は未実施。**

この資料の「必須」「禁止」「合格条件」はこれから実装・検証する契約です。実装済み・互換性実証済みという意味ではありません。付属 `document_validation.json` のPASSはJSON Schemaと例の構造検証だけです。

## 0. 対象・正本・前の説明の修正

互換動作の正本は、Forge Neo `neo`系の特定コミット
`710f1e25fcac84d880cbccf27d11b2e3276e589e` に固定します。
「常に最新」という変動ターゲットにはしません。次の上流変更は別profile/revisionとして認証し、保存済みworkflowを無断更新しません。
比較参考のNeo-Samplerは `925257cf3a97ea203d30d37fb24b3657721c986a`。これは独立した正解データではなく移植例です。テストの期待値は固定Forge正本から作ります。

### 重要な訂正

1. **Animaに一般的なClip Skipを適用する仕様は採用しません。** 固定ForgeのAnima engineは `set_clip_skip` をoverrideしておらず、baseの実装は `pass`。Anima text engineはQwenの最終出力を使用します。Comfyに `clip_layer(-2)` を一律適用すると、互換化ではなく挙動変更になり得ます。[S1][S2][S3]
2. **メタデータにClip Skipがないから2、とは断定しません。** 固定Forgeのinfotextは通常 `is_sd1` の場合だけClip Skipを保存します。SDXLでは設定が作用しても保存されない場合があり、Animaでは設定自体が非適用です。UIの出荷時初期値と、保存画像に適用された実値を区別します。[S4][S5]
3. Neo-Samplerの `rng.py` には `torch.manual_seed` が残っています。「Comfyのsampler関数を置換しない」だけでは実行時の無干渉を保証できません。新実装ではデフォルトRNGへの書込みも禁止します。[S6]
4. 9月23日のQwen embedding変更を、そのままAnimaの差の原因とは扱いません。Animaの呼出側はすでに `embeds` を供給するため、変更されたfallback行を通らない経路があります。実行されたembedding/Qwen/adapter各段階のdtypeと値を検証します。[S1]
5. これまでの画像目視と圧縮WebP上の近似は、正式なStrong ParityのPASSではありません。元のForge/Comfy画像にはnegative promptの空白差も残っています。数値評価用の正本画像は新たに同文・同条件・ロスレスで生成します。

### V1の保証候補範囲

| 範囲 | V1の扱い |
|---|---|
| Anima / SD1.5 / SDXL Base のtxt2img | 実機認証対象。Animaを最初に実装 |
| 基本img2img | 同一初期latentからのsamplingを認証。VAE encodeからの一致は別テスト |
| 21 sampler名 / 17 scheduler名 | 読込み・分類用の登録候補一覧。全組合せの対応保証ではない |
| Hires fix / inpaint / Refiner / Textual Inversion / Anima参照画像 / 動的LoRA extension | 原情報保存・未対応表示まで。V1 Strict実行は停止 |
| SDXLのRF/v-pred派生、Flux、Qwen Image、Krea2など | 構造の拡張余地を残すが、Baseの結果から互換保証を流用しない |
| H3 / LTX / その他video | 互換処理の対象外。インストール・別画像生成後にも通常動作を妨げないことを検証 |

未対応機能を黙って外してtxt2imgに変換することは禁止します。初期認証候補以外のsampling機能は、CPU監査と対応fixtureの実機テストを通した分だけ追加します。

## 1. ForgeGenerationSpec

### 1.1 基本原則

`ForgeGenerationSpec` はJSONへ往復できる設定データです。Tensor、MODEL、CLIP、VAE、Generator、関数、Pythonクラス、絶対ローカルパスを入れません。実行時の重いオブジェクトは別bundleへ分離します。

Pythonではfrozen dataclass相当の扱いとし、子のdict/listも内部で変更しません。ノード間では同一Specを共有できますが、変更時には必ず新しいSpecを生成します。

| 最上位フィールド | 型 | 定義 |
|---|---|---|
| schema_version | 固定文字列 | `1.0.0` |
| profile | object | profile ID、正本repo、正本commit。画像のexporter commitとは別物 |
| policy | enum | `strict` / `exploratory` |
| status | enum | `ready` / `needs_review` / `unsupported` |
| source | object | 原画像名/hash、exporter/version、検証できたcommit、原infotext、順序付きraw_fields |
| requested | JSON Pointer→値 | 保存画像・ユーザーが要求した値。実行時非適用の値も残す |
| effective | GenerationConfig / null | 型変換・モデル別解釈後の設定。候補が組めても未解決事項があればstatusはreadyにしない |
| provenance | JSON Pointer→由来 | metadata_explicit / metadata_implied / manual_input / user_override / profile_default / model_config、確度、確認状態、根拠 |
| unresolved | issue[] | 値不足、asset不明、重複・矛盾など |
| unsupported | issue[] | 未対応のモデル/機能/組合せ |
| warnings | issue[] | 非適用値、環境不一致、未認証条件など |
| extensions | JSON object | 未知フィールド等の保存領域。コードとして実行しない |
| config_hash | SHA256 / null | profileとeffective設定の正規化hash。実行資産や環境の実測確認とは別 |

Pointerは `effective` を根とする `/noise/seed`、`/text/clip_skip` のような形式です。provenanceは親単位でも指定でき、最長一致する子の記録を優先します。Schemaに加えてsemantic validatorがpointerの存在・型・適用可能性を検査します。

### 1.2 effectiveのグループ

| グループ | 主な項目 |
|---|---|
| family / prediction_type / mode | anima・sd15・sdxl / flow・epsilon・v_prediction / txt2img・img2img等 |
| assets[] | role、component、category、相対ファイル名、完全SHA256、元の短縮hash、解決/検証状態 |
| loras[] | asset ID、適用順、model強度、text encoder強度 |
| image | width、height、batch_size |
| text | positive_raw、negative_raw、parser、emphasis、clip_skip、layer_policy、comma backtrack、schedule clock、LoRA方針 |
| noise | seed、CPU/GPU/NV、ENSD、subseed、variation強度、batch seed規則、seed resize、stream規則 |
| sampling | sampler/scheduler、steps/CFG、denoise、img2img step規則/extra noise、Eta/DDIM Eta、churn/tmin/tmax/noise、sigma範囲/rho、Beta α/β、discard、SGM、shift、noise schedule |
| guidance | Skip Early CFG、NGMS、全step適用、batch方針 |
| numeric | MODEL/TE/Anima adapter/sampling/VAEのdtype要求、attention backend要求、参照環境ID |
| sdxl | original/crop/targetの寸法、zero_empty_negative。非SDXLではnull |

詳細の型・必須項目・範囲は `ForgeGenerationSpec.schema.json` を正本とします。JSON Schemaだけでは満たせない契約は以下のsemantic validationで補います。

### 1.3 型・値・由来の規則

- seed、subseed、ENSDは10進**文字列**。Python境界で整数化し、uint64範囲とseed+offset/seed+iの範囲を検証します。JavaScript Number経由で丸めません。乱数seed `-1` は実行Specに入れず、UIで確定seedへ解決して保存します。
- floatは有限値だけ。NaN/Infは禁止。`s_tmax=0` は正本が使う無限大相当の指定としてそのまま保持し、実行時だけ解釈します。
- `sigma_min/max/rho=null` はモデル/profile既定を要求する意味です。元画像の0 sentinelはrequestedに0を残してnullへ正規化します。「情報欠落」はunresolved/provenanceで表し、このnullと混同しません。
- key欠落、0、false、nullを区別します。`if value:` を使った0の取りこぼしは禁止します。
- user override > 明示メタデータ > 検証済み省略規則 > モデル設定/profile候補の順。明示値同士の矛盾は上書きせず停止。profile初期値は画像の実値だと偽らず、影響がある推定値はレビューで確認します。
- `Version: neo-2.29.1` だけからexporter commitを710fへ書き換えません。互換ターゲットcommitと元の生成commitを分離します。
- assetのModule番号から役割を決めません。型・safetensors情報・登録カテゴリ・検証済みhashから解決します。複数候補をfuzzy検索の先頭に決めることは禁止。
- 短縮hashは候補照合用。完全な重み同一性の証明には使いません。Strict実行は明示的に解決した既知loader chainを要求し、正式認証のfixtureでは完全SHA256も必須です。
- 画像寸法を黙って丸めません。V1はAnimaが16の倍数、SD1.5/SDXLが8の倍数を受け付け、さらに実際のMODELの制約を検証します。
- V1のAnima seed resizeは未認証のためStrict停止。通常画像の4D処理を5Dへ無条件転用しません。
- 使わない設定は原値を保存し、Reportで `not_applied` を表示します。無効な組合せと、正本上の意図したno-opを分けます。

### 1.4 Clip Skip / SGMのモデル別確定値

| Family | requested Clip Skip | effective / 実処理 |
|---|---|---|
| Anima | 1/2/4等を保存可能 | `clip_skip=null`、`layer_policy=anima_last`。Forge互換では層を飛ばさない |
| SD1.5 | 1～12 | 指定層 + 正本のfinal layer norm |
| SDXL Base | 1～12 | `effective.clip_skip=max(requested,2)`。L/G両方とpooled出力を別途正しく扱う |

SDXLの非空negative、空negative、`zero_empty_negative`、サイズ/crop/targetの条件ベクトルを独立項目にします。[S5]
SGMはsampler名でもscheduler名でもありません。epsilon/v系では初期noiseの倍率選択、Animaのflow枝では同じsqrt式を適用しません。[S7]

### 1.5 hash / cache

config_hashの入力は `{profile, effective}`。型検証後に整数フィールドをint、floatフィールドをfloat64へ正規化し、-0.0を0.0へ統一します。backendで `json.dumps(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)` を使い、そのUTF-8へSHA256を計算します。frontendは独自のNumber変換やhash再計算をしません。

config_hashは設定一致確認だけです。認証run hashには、実資産hash、MODEL/CLIP patch chain、入出力latent hash、dtype、backend/CUDA/PyTorch/Comfy/extension/profile revision、実際のsigma・RNG traceを別途含めます。

Text cache keyにはprofile、promptの内容、TE/LoRA binding、実dtype、layer方針、SDXL補助条件を含めます。noise cacheにはprofile、shape、batch、seed列、ENSD、variation、入力latentを含めます。cache hitでも実行中の可変Generatorを再利用しません。

## 2. 4ノードの入出力

表示名とclass ID、ポート型を以下で固定します。基本設定はSpecの編集画面だけで変更し、Sampler側に重複seed等を持たせません。

### 2.1 Forge Compat Import / Spec (`ForgeCompatSpec`)

必須入力:
- `source_mode`: `manual / image / infotext / spec_json`
- `profile`: `forge-neo@710f1e25`
- `policy`: `strict / exploratory`
- `payload`: STRING。manual用の構造化値、infotext本文、またはSpec JSON。専用UIで編集するので手入力JSONは必須にしない。
- `overrides_json`: STRING、初期 `{}`。入力時に検査済みのcanonical field overrides。未知キー/不正型は停止。

任意入力: `image_file` STRING。Comfy input配下の選択済みファイル。絶対パス・外部URLは不可。

出力: `spec: FORGE_GENERATION_SPEC`、`report: STRING(JSON)`。

通常の `IMAGE` Tensorをmetadata源にしません。そこには元ファイルのEXIFが保存されないためです。元画像を読み、情報不足をreportする段階と、新workflowを作るfrontend処理を分離します。

### 2.2 Forge Compat Text (`ForgeCompatText`)

必須入力: `spec: FORGE_GENERATION_SPEC`、`clip: CLIP`。
出力: `conditioning: FORGE_CONDITIONING`、`report: STRING(JSON)`。

Textはraw promptを正本parserで解析し、token IDs・weights・mask・必要なprompt scheduleを準備します。AnimaはQwen出力とT5 IDs/weightsを保持し、adapter前後を同じものとして扱いません。モデル側adapterはSamplerのprivate実行経路で処理・追跡します。

**標準CONDITIONINGへ無理に偽装しません。** schedule付きデータを標準KSamplerに入れて黙って最終promptだけへ簡略化する事故を避けるため、V1は専用Samplerへ接続するcustom型です。将来native出力を設ける場合も、静的条件に限る別契約とします。

`FORGE_CONDITIONING` の内部:
- bundle_version、config_hash、family、profile ID、TE/LoRA binding hash
- 正promptのpartごとのweightと `end_at_denoiser_call → conditioning_id`
- 負promptの正本どおりのschedule（正promptのAND規則を機械的に流用しない）
- conditioning_idに対応するtensor/補助metadata
- token/mask/weightのtrace hash、実dtype

Tensorは出力後に上書きせず、consumerが必要に応じてcloneします。bundleをworkflow JSONへ直接serializeせず、Specから再生成します。

### 2.3 Forge Compat Noise (`ForgeCompatNoise`)

必須入力: `spec: FORGE_GENERATION_SPEC`、`model: MODEL`。
任意入力: `input_latent: LATENT`（img2imgだけ必須）。
出力: `latent: LATENT`、`noise: FORGE_NOISE_BUNDLE`、`report: STRING(JSON)`。

MODELのfamily/latent_formatとmodeを先に検査します。AnimaのT=1の画像latentを許し、NestedTensorや動画用MODELをこのノードへ誤接続した場合は説明付きで拒否します。ノード未使用の動画workflowを検査・変更してはいけません。

LATENTにはtxt2imgのゼロlatent、または入力latentのcopyを出します。初期noiseは `samples` に混ぜず別bundleに出します。

`FORGE_NOISE_BUNDLE` の内部:
- bundle_version、config_hash、profile ID、family、shape、dtype、batch/seed列
- 初期noise Tensor（未scale）、初期noise hash、入力latent hash
- 初期noise生成後の各private streamのsnapshot（bytesまたはPhiloxのseed/offset）
- variation、ENSD規則、samplerごとのstep-noise/Brownian生成に必要な情報

**live `torch.Generator` を共有しません。** Samplerはsnapshotから毎回private generatorを再構築します。cache hit、同一bundleの分岐使用、再Queue、cancel/retryで乱数cursorが進んだままになることを禁止します。

### 2.4 Forge Compat Sampler (`ForgeCompatSampler`)

必須入力:
- `spec: FORGE_GENERATION_SPEC`
- `model: MODEL`
- `conditioning: FORGE_CONDITIONING`
- `noise: FORGE_NOISE_BUNDLE`
- `latent_image: LATENT`
- `trace_level`: `none / summary / tensors`、初期summary

出力: `latent: LATENT`、`sigmas: SIGMAS`、`report: STRING(JSON)`。

入力bundleのconfig_hash、MODEL family、実asset binding、shape/dtype、入力latent hashを照合してからGPUへ進みます。違うSpec枝のnoise/conditioningはエラーにします。

処理順:
1. family/predictorと適用可能設定を確定。
2. Automatic scheduler、sampler固有既定値、強制discard規則を解決。
3. 正本の手順でsigma列を生成。requested discardはsampler固有discardとORして1回だけ適用。必要ならN+1で生成してから最後から2番目を除去。[S8]
4. initial noise scalingを1回だけ適用。
5. private RNG/Brownian sampler、CFG/schedule/negative skip、実行単位のモデルadapterでsampling。
6. LATENT、実sigma列、実行reportを出す。

`len(sigmas)==requested_steps+1` を全schedulerに強制しません。Betaの丸め・重複排除、restart等の実呼出しはtraceに記録します。SIGMAS出力は基礎列、restart等による追加evaluationはreportのcall traceです。

prompt scheduleはUIのstep数や単純なsigma百分率ではなく、固定Forgeのdenoiser呼出し数・境界比較を再現します。samplerの二次評価を含めます。

img2imgは `forge_scaled` / `exact_steps` を分離し、正本の整数化とsigma sliceを再現します。denoise=0を勝手にidentity扱いへ変更せず、正本上不成立の組合せは明示的に拒否します。[S9]

### 2.5 接続図

```text
画像Drag&Drop / 手動編集
        ↓
Forge Compat Import / Spec ── SPEC ─┬──────────────┬─────────────────┐
                                  │              │                 │
標準CLIP Loader → 明示LoRA → Forge Compat Text     │                 │
                                  │ conditioning │                 │
標準Model Loader → 明示LoRA ────────┼────────→ Forge Compat Noise    │
             │                    │              │ LATENT + NOISE  │
             └────────────────────┴──────────────┴────→ Forge Compat Sampler
                                                         │ LATENT
標準VAE Loader ─────────────────────────────────────→ VAEDecode → SaveImage
```

4ノードは4種類の追加classという意味です。標準loader、LoRA loader、VAE Decode/Encode、SaveImageは別に使用します。LoRAはImporterが生成する**明示loader経路で1回だけ**適用し、Textは構文を条件文字列から分離、Samplerは自動ロードしません。手動接続で使う場合も同じです。未知のloader/patch chainはStrict認証外として検出します。

## 3. Importer / frontend仕様

標準のComfy workflow/API promptを含む画像は、Forge形式のinfotextも併記されていても**native経路を優先**します。単に `Steps:` があるからForge扱いにしません。

PNGの文字metadata、JPEG/WebPのUserComment、UTF-16のBOM/byte order、Exif IFD、RIFF odd-byte paddingを扱います。fixtureとして添付済みWebP 6枚を登録しました。これは実ファイルからmetadataを抽出した記録で、生成試験結果ではありません。

フロントエンドの専用Import panel / menuは提供します。canvas上の通常Drag&Drop統合は設定 `ForgeCompat.Import.enable_canvas_drop` を有効にした場合だけ使用。無効時はnativeのhandleFileに一切介入しません。

統合有効時にfile handlerのchainが必要なfrontend版では、Forgeとして解釈できるファイルだけを処理し、その他は元handlerへ一度だけ委譲します。既存graphを先にclearしません。事前検証、asset解決、レビュー後に新しいworkflow/tabを開きます。失敗・cancelは元graphを保持します。API/version不一致なら統合だけ無効化し、panelからのImportを維持します。

Importは生成設定の復元であって、外部資産の自動取得・Python実行・extension導入ではありません。未登録node classの自動作成も禁止します。

入力上限: 元画像128 MiB、展開後metadata合計16 MiB、1 prompt 100,000文字、IFD再帰深さ8。offsetの範囲外、再訪IFD、サイズ超過は停止。relative pathは許可カテゴリ内へ正規化し、`..`・絶対パス・UNC・URL等を拒否します。

### Reportの共通形式

全nodeのreportはJSONのSTRINGです。schema_version、node ID、config_hash、status、requested/applied/not_applied、inferred/unresolved/unsupported、actual asset/dtype/shapeの要約、trace identifiersを返します。

Samplerの `status=completed` は画像生成完了を意味するだけです。`qualification` は別欄で `not_evaluated / exact / strong / partial / failed` とし、比較結果がない実行をstrongにしません。

代表エラー: `INVALID_SPEC`, `NEEDS_REVIEW`, `ASSET_MISSING`, `ASSET_AMBIGUOUS`, `MODEL_FAMILY_MISMATCH`, `UNSUPPORTED_COMBINATION`, `SPEC_MISMATCH`, `LATENT_MISMATCH`, `DTYPE_NOT_APPLIED`, `UNVERIFIED_PATCH_CHAIN`, `NATIVE_METADATA_PRIORITY`, `IMPORT_LIMIT_EXCEEDED`。

## 4. 無干渉・数値安全性

禁止:
- comfy.sample/comfy.samplers/torchの関数やmodule globalを置換。
- `torch.manual_seed` / `torch.cuda.manual_seed_all` でデフォルトRNGを変更。
- run間で可変scheduler CTX、noise callable、torch proxyを共有。
- cloneしたwrapperの背後で共有されるMODEL/CLIP本体へ無条件 `.to(dtype)` や属性/hookを書込み。
- import時のpip install、ネットワーク取得、旧workflowの一括変更。
- OOM時に勝手にタイル生成、解像度/steps/dtype変更へ切替。

必要な演算精度はrun-local adapter/operationsで適用するか、対応していなければ明示停止します。重みのロード・offload等、Comfy Coreが管理する配置の変化は許されますが、extensionが意味的な重み/optionsを汚染しないことを差分テストします。単なる `model.clone()` / `clip.clone()` の呼出しだけを無干渉の証明にしません。

キャンセル・例外でもrun-local参照と一時cacheを破棄します。GCでしか回収できない参照cycleを作りません。Thread/context間のprivate state独立性をCPU試験で確認してからGPUへ進みます。

移植ソースを利用する場合は元の著作権表示とAGPL条件等を保持します。今回の資料は新しいカスタムノードの配布物ではなく設計資料です。

## 5. テストマトリクス

機械可読の全44試験定義は `test_matrix.json`。全件 `NOT_RUN` であり、パラメーター展開後のrender数とは異なります。以下は中心となる組合せです。

### 5.1 共通sampling実機比較

| ID | Sampler / Scheduler | Anima | SD1.5 | SDXL | 変更条件 |
|---|---|---|---|---|---|
| G01 | Euler / Simple | 必須 | 必須 | 必須 | seed 0 / 42 / 1895162280 |
| G02 | Euler / Beta | 必須 | 必須 | 必須 | seed1895162280 |
| G03 | ER-SDE / Beta | 必須 | 必須 | 必須 | seed 0 / 42 / 1895162280 |
| G04 | Euler a / Simple | 必須 | 必須 | 必須 | ENSD0 / 31337 |
| G05 | DPM++2M / Karras | 必須 | 必須 | 必須 | 決定論sampling基準 |
| G06 | DPM++2M SDE / Exponential | 必須 | 必須 | 必須 | Brownian経路 |
| G07 | DPM2 / Karras | 必須 | 必須 | 必須 | sampler固有discardと二次評価 |

G01～G07は36組のForge↔Comfy比較。既存の同一基準renderは再利用します。さらにGPU/NV乱数、batch2、同一初期latentのimg2imgを代表seedで確認します。まずCPU試験を通し、不合格経路について無闇に全GPU組合せを回しません。

### 5.2 モデル専用

| 対象 | 必須検証 |
|---|---|
| Anima | Clip Skip1/2/4がForgeと同様no-op、Qwen/T5 ids/weights/masks、adapter前後、TE/adapter実dtype、Turbo LoRA1回適用、SGMのflow no-op |
| SD1.5 | Clip Skip1/2/4とfinal norm、74/75/76・149/150/151 token境界、comma backtrack/BREAK、SGM ON/OFF |
| SDXL | Clip Skip1→2/2/4、L/G長さ差とconcat、pooled出力、original/crop/target条件、空negativeをゼロ化する設定ON/OFF |

全familyで重み構文・prompt editing・AND・negative skipをまずCPU/conditioning段階で検証し、代表caseだけGPUへ進みます。「Original / Ignore」等の名称から期待効果を決めず、モデル別の正本挙動を期待値にします。

### 5.3 fixture固定

Animaの既存画像は `anima-Luc_v3D`、短縮model hash `d6ee91d677`、Turbo LoRA短縮hash `1b55e40bdb1d`、1024×1344 / 10 steps / CFG1.5 を開始条件とします。モデル・LoRAの完全SHA256、TE/VAE実ファイル、実行環境の登録がGPU比較前に必要です。

SD1.5は `SD15_BASE_EPS`、SDXLは `SDXL_BASE_EPS` のfixture IDを固定します。実モデルがこの会話では指定されていないため、架空のファイル名/hashを埋めません。GPU前にローカル資産へ結び、両環境で同じファイルを使用します。SD1.5は512×768 / 20 steps / CFG7、SDXLは1024×1024 / 20 steps / CFG5を試験設定とします。

一般samplingテストの共通prompt:
`A red ceramic teapot on a wooden table, a white cup on the right, soft window light, plain background.`
negative:
`blurry, low quality, text, watermark`

ユーザーのAnima画像は別の回帰anchorです。元Forge infotextを文字列単位で正本にし、LoRA構文分離以外の空白や末尾句読点を「見た目の整形」で変更しません。

### 5.4 合格基準

**Exact Inputs:** 同じ型・shape・依存ライブラリ条件でtoken IDs、mask、weights、初期/step noise、CPU sigma列が一致。seedが同じという理由だけでは合格にしません。

**Strong Parity:** 以下の全条件を各fixture比較で満たすことを目標とします。
- final latent相対L2 `||test-ref|| / max(||ref||, 1e-12) <= 0.01`
- ロスレスRGB（0..1）のSSIM >= 0.98
- ロスレスRGBのrange-normalized RMSE `sqrt(mean((test-ref)^2)) <= 0.02`
- 顔・手・ポーズ・構図・物体の位置に重大な差がないことを人が確認

SSIMはRGBの各channel、11×11窓、Gaussian sigma1.5、population covariance、data_range1で計算して平均します。画像位置合わせ、resize、crop、色補正、良いseedの選別で合格させません。全体平均で失敗caseを隠しません。

これらは本仕様で定める**目標閾値**であり、実測から達成済みと判定した値ではありません。満たせなければPARTIAL/FAILとし、テスト後に閾値だけを下げません。

**Exact Output:** final latentと非圧縮RGB配列が完全一致した場合だけ付与。PNG等のファイルhashはmetadata差があるため判定対象外です。

出力比較前にForge側の自己再現性を確認します。Forge/Comfyのcommit、Comfy frontend版、PyTorch/CUDA、GPU、attention、量子化、TE/adapter/VAE/計算dtype、cond/uncond batch経路、LoRAの適用順を記録。異なるハードウェア・演算backend・未確認派生モデルまで結果を一般化しません。

### 5.5 無干渉・UI Gate

H3/LTXは「拡張なし → インストール済み未使用 → Compat画像生成後 → Compatエラー後」のnative workflowを比較します。Nested streamが全て保たれ、実行成功・queue残りなしを確認。元のnative自己再現がbit-stableならpixel/latent一致も要求します。元から不安定な条件を使って「同じはず」と合格させません。

Importerはネイティブworkflow優先、非Forgeファイルへのdelegate、保存/再読込、seed大整数、異常metadata、既存graphの保護、キャンセル、設定OFF時無介入をブラウザーで検証します。

## 6. 実装順と完了条件

1. Spec/validation/canonical hash/metadata抽出・asset候補report。CPU Gate。
2. private RNGとsampling。解析用denoiserで正本比較。
3. Anima Text/adapter → Anima実機Gate。既存画像を再生成基準に固定。
4. SD1.5/SDXL Text、専用fixture実機Gate。
5. frontend graph生成/Drag&Drop、無干渉Gate、保存再読込。

仕様変更が必要ならversionと理由を記録します。1段階のFAILを次のUIで隠しません。認証manifestは通過したモデル・operation・sampler/scheduler・精度・環境の範囲だけを公表します。4種類のnodeが出現しただけでは完成にしません。

## 7. 今回作成・確認したもの

- JSON Schema定義と3 familyの例: 構造検証済み。
- 添付済みWebP 6枚: 実ファイルのSHA256、infotext、存在するAPI promptを抽出済み。画像原本は変更していません。
- 44試験定義: 作成のみ。製品のCPU/GPU/browser試験は未実行。
- コード実装、Comfyへの導入、Git commit/push/release: 未実施。

## 参照ソース（固定コミット）

S1: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/backend/text_processing/anima_engine.py
S2: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/backend/diffusion_engine/anima.py
S3: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/backend/diffusion_engine/base.py
S4: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/modules/processing.py
S5: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/backend/diffusion_engine/sdxl.py
S6: https://github.com/wangjue520/ComfyUI-Neo-Sampler/blob/925257cf3a97ea203d30d37fb24b3657721c986a/neo_core/rng.py
S7: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/backend/modules/k_prediction.py
S8: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/modules/sd_samplers_kdiffusion.py
S9: https://github.com/Haoming02/sd-webui-forge-classic/blob/710f1e25fcac84d880cbccf27d11b2e3276e589e/modules/sd_samplers_common.py
