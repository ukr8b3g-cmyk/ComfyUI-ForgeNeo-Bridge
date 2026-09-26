"""Use Comfy's public node implementations with the currently edited inputs."""
from .spec import BridgeError, canonical
from .workflow_adapter import ForgeTextInput, _clip_skip


def sample_comfy(model, positive, negative, settings, sampler, latent):
    import nodes as core
    from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES

    if not isinstance(positive, ForgeTextInput) or not isinstance(negative, ForgeTextInput):
        raise BridgeError('INVALID_SPEC','Connect ForgeNeo Bridge Text Encode nodes')
    values = settings.values
    # These are naming aliases, not substitutes for unsupported schedules.
    scheduler = {'automatic':'normal','uniform':'normal','ddim':'ddim_uniform'}.get(sampler['scheduler'],sampler['scheduler'])
    sampler_name = {'unipc':'uni_pc'}.get(sampler['sampler_name'],sampler['sampler_name'])
    if sampler_name not in SAMPLER_NAMES or scheduler not in SCHEDULER_NAMES:
        raise BridgeError('COMFY_UNSUPPORTED_SAMPLING','Choose a sampler and scheduler supported by ComfyUI, or enable Forge compatibility')
    seed = str(sampler['seed'])
    if not seed.isdecimal() or not 0 <= int(seed) <= 0xffffffffffffffff:
        raise BridgeError('INVALID_SPEC','Seed must be an unsigned 64-bit integer')
    if latent is None:
        if values['mode'] == 'img2img':
            raise BridgeError('LATENT_MISMATCH','Connect the encoded image latent for img2img')
        latent, = core.EmptyLatentImage().generate(values['width'],values['height'],values['batch_size'])
    # Auto leaves the loaded model's defaults intact. Clip Skip uses the same
    # cloned CLIP as CLIPSetLastLayer; never mutate the shared loader output.
    skip = _clip_skip(values.get('clip_skip','auto'),None)
    def encode(text_input):
        clip = text_input.clip
        if skip is not None:
            clip, = core.CLIPSetLastLayer().set_last_layer(clip,-skip)
        conditioning, = core.CLIPTextEncode().encode(clip,text_input.text)
        return conditioning
    result = core.KSampler().sample(model,int(seed),sampler['steps'],sampler['cfg'],
                                   sampler_name,scheduler,encode(positive),encode(negative),
                                   latent,denoise=sampler['denoise'])
    info = {'node':'ForgeNeoBridgeKSampler','execution':'comfy','scheduler':scheduler,
            'note':'Core CLIPTextEncode / CLIPSetLastLayer / KSampler. Forge-only settings are retained but not applied.'}
    return {'ui':{'text':[canonical(info)]},'result':result}
