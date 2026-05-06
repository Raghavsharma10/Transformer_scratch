def import_module(self, name):
        """Import a module into the bridge."""
        if name not in self._objects:
            module = _import_module(name)
            self._objects[name] = module
            self._object_references[id(module)] = name
        return self._objects[name]