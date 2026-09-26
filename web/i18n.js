/** Follow the Comfy UI preference, rather than the operating-system language. */
export function language(app) {
    const locale = app?.ui?.settings?.getSettingValue?.('Comfy.Locale')
        ?? app?.extensionManager?.setting?.get?.('Comfy.Locale')
        ?? globalThis.document?.documentElement?.lang ?? 'en';
    return /^ja(?:-|$)/i.test(locale) ? 'ja' : 'en';
}
export function tr(app, ja, en) { return language(app) === 'ja' ? ja : en; }

export const SETTINGS_LABELS = {
    forge_compatibility:['Forge Neo互換','Forge compatibility'],
    sampling_adjustments:['サンプリング差分補正','Sampling adjustments'],
    family:['モデル種別','Model family'], mode:['生成モード','Mode'],
    width:['幅','Width'], height:['高さ','Height'], batch_size:['枚数','Batch size'],
    clip_skip:['Clip Skip','Clip Skip'], ensd:['ENSD','ENSD'],
    emphasis:['強調方式','Emphasis'], comma_padding_backtrack:['カンマ後退数','Comma backtrack'],
    rng:['乱数方式','RNG'], subseed:['サブシード','Subseed'], subseed_strength:['変化量','Variation strength'],
    seed_resize_width:['シード基準幅','Seed resize width'], seed_resize_height:['シード基準高さ','Seed resize height'],
    eta_ancestral:['Eta ancestral','Eta ancestral'], eta_ddim:['Eta DDIM','Eta DDIM'],
    s_churn:['S churn','S churn'], s_tmin:['S tmin','S tmin'], s_tmax:['S tmax','S tmax'], s_noise:['S noise','S noise'],
    sigma_min:['Sigma最小','Sigma min'], sigma_max:['Sigma最大','Sigma max'], rho:['Rho','Rho'],
    beta_alpha:['Beta alpha','Beta alpha'], beta_beta:['Beta beta','Beta beta'],
    discard_penultimate:['末尾直前sigma除外','Discard penultimate'],
    sgm_noise_multiplier:['SGMノイズ倍率','SGM noise multiplier'], shift:['Shift','Shift'],
    skip_early_cfg:['初期CFG省略','Skip early CFG'], ngms:['NGMS','NGMS'], ngms_all_steps:['全ステップNGMS','NGMS all steps'],
    img2img_step_mode:['ステップ計算','Step mode'], img2img_extra_noise:['追加ノイズ','Extra noise'],
    original_width:['元の幅','Original width'], original_height:['元の高さ','Original height'],
    target_width:['目標幅','Target width'], target_height:['目標高さ','Target height'],
    crop_x:['クロップX','Crop X'], crop_y:['クロップY','Crop Y'],
    zero_empty_negative:['空のネガティブをゼロ化','Zero empty negative'], source_json:['読み込み情報','Source information'],
};
