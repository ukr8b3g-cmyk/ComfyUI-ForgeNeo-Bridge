import concurrent.futures,copy,torch,pytest
from fnb.bridge.rng import ImageRNG,RNGConfig
from fnb.bridge.noise import freeze_state,restore_state,seeds_for
from fnb.bridge.spec import default_document

@pytest.mark.parametrize('source',['CPU','NV'])
@pytest.mark.parametrize('shape',[(4,8,6),(16,1,8,6)])
@pytest.mark.parametrize('ensd',[0,31337])
def test_private_replay(source,shape,ensd):
    before=torch.random.get_rng_state().clone()
    rng=ImageRNG(RNGConfig(source,'cpu',ensd),shape,[42,43],subseeds=[0,1])
    initial=rng.next();state=freeze_state(rng);next1=rng.next()
    other=ImageRNG(RNGConfig(source,'cpu',ensd),shape,[42,43],subseeds=[0,1]);restore_state(other,state)
    assert torch.equal(next1,other.next())
    assert torch.equal(before,torch.random.get_rng_state())
    assert initial.dtype==torch.float32 and initial.shape==(2,*shape)
    assert all(not isinstance(v,torch.Generator) for v in state)

@pytest.mark.parametrize('source',['CPU','NV'])
def test_batch_matches_individual(source):
    def r(seeds):return ImageRNG(RNGConfig(source,'cpu'),(4,8,8),seeds)
    batch=r([0,1]);a=r([0]);b=r([1])
    for _ in range(4):assert torch.equal(batch.next(),torch.cat((a.next(),b.next())))

def test_interleaving_threads():
    def work(seed):
        rng=ImageRNG(RNGConfig('CPU','cpu'),(4,8,8),[seed]);return [rng.next() for _ in range(5)]
    before=torch.random.get_rng_state().clone()
    sequential=[work(s) for s in (42,1337)]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:parallel=list(pool.map(work,(42,1337)))
    assert all(torch.equal(a,b) for aa,bb in zip(sequential,parallel) for a,b in zip(aa,bb))
    assert torch.equal(before,torch.random.get_rng_state())

def test_variation_seed_rule():
    cfg=default_document()['effective'];cfg['image']['batch_size']=2
    cfg['noise']['subseed_strength']=.4
    assert seeds_for(cfg)[0]==(1895162280,1895162280)
