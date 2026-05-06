def create_module(self, spec):
        """Creates the module, and also insert it into sys.modules, adding this onto py2 import logic."""
        mod = sys.modules.setdefault(spec.name, types.ModuleType(spec.name))
        # we are using setdefault to satisfy https://docs.python.org/3/reference/import.html#loaders
        return mod