import json,struct,zlib,pytest
from fnb.bridge.metadata import read_metadata,_tiff,Collector
from fnb.bridge.infotext import import_infotext,parse_loras,split_fields
from fnb.bridge.spec import BridgeError

TEXT='a teapot\nNegative prompt: blurry\nSteps: 10, Sampler: ER SDE, Schedule type: Beta, CFG scale: 1.5, Seed: 9007199254740993, Size: 1024x1344, RNG: CPU, ENSD: 0, Eta: 0, Sigma noise: 0, Beta schedule alpha: 0.6, Beta scheduler beta: 0.6, SGM noise multiplier: False, Version: neo-2.29.1'
def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
def png(values):return b'\x89PNG\r\n\x1a\n'+b''.join(chunk(b'tEXt',k.encode()+b'\0'+v.encode()) for k,v in values)+chunk(b'IEND',b'')
def tiff(text,bom=False):
    data=b'UNICODE\0'+((b'\xff\xfe'+text.encode('utf-16-le')) if bom else text.encode('utf-16-be'))
    # piexif/Forge layout: Exif SubIFD omits the next-IFD pointer.
    return b'MM\x00*'+struct.pack('>I',8)+struct.pack('>HHHIII',1,0x8769,4,1,26,0)+struct.pack('>HHHII',1,0x9286,7,len(data),40)+data

def webp(exif,odd=False):
    data=(b'JUNK'+struct.pack('<I',1)+b'X\0' if odd else b'')+b'EXIF'+struct.pack('<I',len(exif))+exif+(b'\0' if len(exif)%2 else b'')
    return b'RIFF'+struct.pack('<I',len(data)+4)+b'WEBP'+data

@pytest.mark.parametrize('kind',['png','webp','webp_bom','jpeg'])
def test_formats(kind):
    e=tiff(TEXT,kind=='webp_bom')
    data=png([('parameters',TEXT)]) if kind=='png' else b'\xff\xd8\xff\xe1'+struct.pack('>H',len(e)+8)+b'Exif\0\0'+e+b'\xff\xd9' if kind=='jpeg' else webp(e,True)
    result=read_metadata(data);assert result['kind']=='forge';assert result['metadata']['parameters']==TEXT

def test_native_priority():
    assert read_metadata(png([('parameters',TEXT),('workflow','{"nodes":[]}')]))['kind']=='native'

def test_explicit_zero_seed_aliases():
    doc=import_infotext(TEXT).document();c=doc['effective']
    assert c['noise']['seed']=='9007199254740993'
    assert c['sampling']['eta_ancestral']==0 and c['sampling']['s_noise']==0
    assert c['noise']['ensd']=='0' and c['sampling']['sgm_noise_multiplier_requested'] is False
    assert doc['source']['verified_exporter_commit'] is None
    assert doc['status']=='needs_review'

def test_conflicting_alias():
    with pytest.raises(BridgeError,match='Conflicting'):import_infotext(TEXT+', Beta scheduler alpha: 0.9')

def test_module_order_unknown():
    d=import_infotext(TEXT+', Module 1: arbitrary_vae, Module 2: arbitrary_encoder').document()
    assert all(a['role_hint'] is None for a in d['extensions']['raw_assets'])

def test_lora_static_and_single():
    cleaned,ls=parse_loras('a <lora:name:0.3:0.7> <lora:b:te=0.5:unet=1>')
    assert cleaned=='a  ' and ls[0]['strength_model']==.7 and ls[0]['strength_text_encoder']==.3
    assert ls[1]['strength_text_encoder']==.5
    with pytest.raises(BridgeError):parse_loras('<lora:x:1@0.5>')

def test_literal_prompt_whitespace():
    text=TEXT.replace('a teapot','  a  teapot ')
    assert import_infotext(text).config['text']['positive_raw']=='  a  teapot '

def test_hires_not_silently_dropped():
    d=import_infotext(TEXT+', Hires upscale: 2').document();assert d['status']=='unsupported'

@pytest.mark.parametrize('data',[b'bad',png([('parameters',TEXT)])[:-2],webp(tiff(TEXT))[:-1],b'\xff\xd8\xff\xe1\xff\xffx'])
def test_truncated_and_malformed(data):
    with pytest.raises(BridgeError):read_metadata(data)

def test_cycle():
    data=b'MM\x00*'+struct.pack('>I',8)+struct.pack('>HI',0,8)
    with pytest.raises(BridgeError):_tiff(data,Collector())

def test_quote_fields():
    assert split_fields('Model: "name, spaces", Extra: {"a": 1, "b": 2}')[0]['value']=='name, spaces'

def test_itxt_and_ztxt():
    raw=b'parameters\0\1\0\0\0'+zlib.compress(TEXT.encode())
    p=b'\x89PNG\r\n\x1a\n'+chunk(b'iTXt',raw)+chunk(b'IEND',b'')
    assert read_metadata(p)['metadata']['parameters']==TEXT
    p=b'\x89PNG\r\n\x1a\n'+chunk(b'zTXt',b'parameters\0\0'+zlib.compress(TEXT.encode()))+chunk(b'IEND',b'')
    assert read_metadata(p)['kind']=='forge'
