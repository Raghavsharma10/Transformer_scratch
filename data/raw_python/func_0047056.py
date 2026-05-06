def unload_module(self, module_name):
        """Unload the specified module, if it is loaded."""
        module = self.loaded_modules.get(module_name)
        if not module:
            _log.warning("Ignoring request to unload non-existant module '%s'",
                         module_name)
            return False

        module.stop(reloading=False)
        del self.loaded_modules[module_name]
        self.module_ordering.remove(module_name)
        return True