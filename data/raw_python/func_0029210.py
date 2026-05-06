def _import_public_names(module):
    "Import public names from module into this module, like import *"
    self = sys.modules[__name__]
    for name in module.__all__:
        if hasattr(self, name):
            # don't overwrite existing names
            continue
        setattr(self, name, getattr(module, name))