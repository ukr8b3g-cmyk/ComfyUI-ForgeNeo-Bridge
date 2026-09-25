"""Per-run torchsde interval namespace. Never patches torchsde._randn globally.

Forge replaces torchsde's module-level _randn. Here the installed 0.2.6 interval
implementation is instantiated in a PRIVATE namespace with the same RNG callback.
Only trusted installed library source is compiled; no metadata/code is evaluated.
"""
from __future__ import annotations
import inspect
import torch
from .rng import randn_local
from .spec import BridgeError


def make_brownian(x, sigmas, seeds, cfg):
    try:
        import torchsde
        from torchsde._brownian import brownian_interval
    except ImportError as exc:
        raise BridgeError('DEPENDENCY_MISSING', 'DPM++ SDE needs ComfyUI\'s torchsde dependency; nothing was installed.') from exc
    if getattr(torchsde, '__version__', '') != '0.2.6':
        raise BridgeError('UNSUPPORTED_COMBINATION', 'Private Brownian adapter currently audited for torchsde 0.2.6 only.')
    # Each run owns its functions/classes and RNG callback. No sys.modules write.
    ns = {'__name__': 'torchsde._brownian._forge_bridge_interval', '__package__': 'torchsde._brownian'}
    exec(compile(inspect.getsource(brownian_interval), brownian_interval.__file__, 'exec'), ns)
    ns['_randn'] = lambda size, dtype, device, seed: randn_local(cfg, seed, size).to(device=device, dtype=dtype)
    t0, t1 = sigmas[sigmas > 0].min(), sigmas.max()
    if len(seeds) != x.shape[0]: raise BridgeError('LATENT_MISMATCH', 'Brownian seed batch mismatch')
    # Exact BrownianTree defaults from torchsde 0.2.6 derived.py.
    intervals = [ns['BrownianInterval'](t0=t0, t1=t1, size=x[0].shape, dtype=x.dtype, device=x.device,
                                      entropy=int(seed), tol=1e-6, pool_size=24, halfway_tree=True, W=None)
                 for seed in seeds]
    return BrownianStream(intervals, ns)


class BrownianStream:
    """Owns, and deterministically releases, the library's cyclic interval tree."""
    def __init__(self, intervals, namespace):
        self.intervals, self.namespace = intervals, namespace
        self.closed = False

    def __call__(self, sigma, sigma_next):
        if self.closed:
            raise BridgeError('RNG_CLOSED', 'Brownian stream was already released')
        a, b = torch.as_tensor(sigma), torch.as_tensor(sigma_next)
        lo, hi, sign = (a, b, 1) if a < b else (b, a, -1)
        return torch.stack([interval(lo, hi) for interval in self.intervals]).to(device=a.device, dtype=a.dtype) * sign / (b - a).abs().sqrt()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for root in self.intervals:
            pending, seen = [root], set()
            while pending:
                node = pending.pop()
                if id(node) in seen:
                    continue
                seen.add(id(node))
                for key in ('_left_child', '_right_child'):
                    child = getattr(node, key, None)
                    if child is not None:
                        pending.append(child)
                    if hasattr(node, key):
                        setattr(node, key, None)
                node._parent = node._top = None
            cache = getattr(root, '_increment_and_space_time_levy_area_cache', None)
            if cache is not None:
                cache.clear()
            root._w_h = root._last_interval = None
        self.intervals.clear()
        # Functions retain their private globals. Clearing releases the run RNG.
        self.namespace.clear()
