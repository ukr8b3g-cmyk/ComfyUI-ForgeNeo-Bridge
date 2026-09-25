"""ComfyUI-ForgeNeo-Bridge: explicit, run-local Forge compatibility."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = './web'
__version__ = '1.0.0'
__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS','WEB_DIRECTORY']

# Register only our own HTTP namespace. No engine/global sampling patches.
try:
    from server import PromptServer
except ImportError:
    PromptServer = None
if PromptServer is not None and hasattr(PromptServer,'instance'):
    from .bridge.server import register_routes
    register_routes(PromptServer.instance.routes)
