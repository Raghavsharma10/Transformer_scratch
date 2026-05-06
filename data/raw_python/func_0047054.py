def reload_module(self, module_name):
        """Reloads the specified module without changing its ordering.

        1. Calls stop(reloading=True) on the module
        2. Reloads the Module object into .loaded_modules
        3. Calls start(reloading=True) on the new object
        
        If called with a module name that is not currently loaded, it will load it.

        Returns True if the module was successfully reloaded, otherwise False.
        """
        module = self.loaded_modules.get(module_name)
        if module:
            module.stop(reloading=True)
        else:
            _log.info("Reload loading new module module '%s'",
                         module_name)
        success = self.load_module(module_name)
        if success:
            _log.info("Successfully (re)loaded module '%s'.", module_name)
        elif module:
            _log.error("Unable to reload module '%s', reusing existing.",
                       module_name)
        else:
            _log.error("Failed to load module '%s'.", module_name)
            return False
        self.loaded_modules[module_name].start(reloading=True)
        return success