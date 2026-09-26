import copy,json,pytest
from fnb.bridge.spec import *

@pytest.mark.parametrize('family',['anima','sd15','sdxl'])
def test_schema_examples(family):
    import jsonschema
    d=finalize(default_document(family)).document()
    jsonschema.Draft202012Validator(schema()).validate(d)
    assert d['status']=='needs_review'
    assert finalize(d).config_hash==d['config_hash']

@pytest.mark.parametrize('path,value',[
 ('/noise/seed',42),('/noise/seed','-1'),('/noise/seed','18446744073709551616'),
 ('/sampling/cfg',float('nan')),('/sampling/steps',True),('/sampling/steps',0),
 ('/image/width',1025),('/sampling/beta_alpha',0),('/text/emphasis','unknown'),('/image/batch_size',0)])
def test_invalid(document,path,value):
    with pytest.raises(BridgeError):finalize(document,{path:value})

def test_normalization_and_immutable(document):
    a=finalize(document,{'/sampling/cfg':2});b=finalize(document,{'/sampling/cfg':2.0})
    assert a.config_hash==b.config_hash
    c=a.config;c['sampling']['cfg']=10
    assert a.config['sampling']['cfg']==2.0
    assert document['effective']['sampling']['cfg']==1.5

def test_recomputes_forged_ready():
    d=default_document();d['status']='ready';d['unresolved']=[]
    s=finalize(d);d=s.document();d['unresolved']=[]
    with pytest.raises(BridgeError,match='NEEDS_REVIEW'):GenerationSpec(canonical(d)).require_executable()

@pytest.mark.parametrize('name',['../x','/x','C:\\x','\\\\host\\file','a/../b','a//b','https://bad','./file'])
def test_path_reject(name):
    with pytest.raises(BridgeError):safe_relative(name)

def test_large_seed(document):
    d=finalize(document,{'/noise/seed':'18446744073709551615','/noise/ensd':'0'})
    assert d.config['noise']['seed']=='18446744073709551615'
    with pytest.raises(BridgeError):finalize(document,{'/noise/seed':'18446744073709551615','/noise/ensd':'1'})

@pytest.mark.parametrize('value',[1,2,4])
def test_anima_clip_noop(document,value):
    # Overrides normalize before family constraints are applied.
    d=finalize(document,{'/text/clip_skip':value})
    assert d.config['text']['clip_skip'] is None
    assert d.document()['requested']['/text/clip_skip']==value

@pytest.mark.parametrize('value',[1,2,4])
def test_sdxl_clip_minimum(value):
    d=finalize(default_document('sdxl'),{'/text/clip_skip':value})
    assert d.config['text']['clip_skip']==max(2,value)

@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_duplicate_and_nonfinite_json(text):
    with pytest.raises(BridgeError):parse_json(text)

def test_policy_and_unknown_fields(document):
    document['unknown']=1
    with pytest.raises(BridgeError):finalize(document)

def test_unimplemented_preserved(document):
    d=finalize(document,{'/sampling/sampler':'plms'})
    assert d.config['sampling']['sampler']=='plms'
    with pytest.raises(BridgeError,match='UNSUPPORTED'):d.require_executable()


@pytest.mark.parametrize('name',['ddim','unipc','euler_cfg_pp','euler_ancestral_cfg_pp','dpmpp_2m_cfg_pp'])
def test_shared_solver_is_usable_without_removing_other_validation(document,name):
    spec=finalize(document,{'/sampling/sampler':name})
    assert spec.require_executable()['sampling']['sampler']==name
    document['effective']['assets']=[]
    with pytest.raises(BridgeError):finalize(document,{'/sampling/sampler':name}).require_executable()


def test_imported_unsupported_information_is_not_discarded(document):
    document['unsupported']=[{'code':'UNSUPPORTED_COMBINATION','path':'/extensions','message':'Unknown extension effects'}, {'code':'UNSUPPORTED_COMBINATION','path':'/text','message':'Textual inversion'}]
    spec=finalize(document)
    assert len(spec.document()['unsupported'])==2
    with pytest.raises(BridgeError,match='UNSUPPORTED'):spec.require_executable()
