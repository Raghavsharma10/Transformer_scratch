def load_module(self, module_name):
        """Attempts to load the specified module.

        If successful, .loaded_modules[module_name] will be populated, and
        module_name will be added to the end of .module_ordering as well if
        it is not already present. Note that this function does NOT call
        start()/stop() on the module - in general, you don't want to call
        this directly but instead use reload_module().

        Returns True if the module was successfully loaded, otherwise False.
        """
        if module_name in self.currently_loading:
            _log.warning("Ignoring request to load module '%s' because it "
                         "is already currently being loaded.", module_name)
            return False

        try: # ensure that currently_loading gets reset no matter what
            self.currently_loading.add(module_name)
            if self.loaded_on_this_event is not None:
                self.loaded_on_this_event.add(module_name)

            # Force the module to actually be reloaded
            try:
                _temp = reload(importlib.import_module(module_name))
            except ImportError:
                _log.error("Unable to load module '%s' - module not found.",
                           module_name)
                return False
            except SyntaxError:
                _log.exception("Unable to load module '%s' - syntax error(s).",
                           module_name)
                return False

            if not hasattr(_temp, "module"):
                _log.error("Unable to load module '%s' - no 'module' member.",
                           module_name)
                return False

            module = _temp.module
            if not issubclass(module, Module):
                _log.error("Unable to load module '%s' - it's 'module' member "
                           "is not a kitnirc.modular.Module.", module_name)
                return False

            self.loaded_modules[module_name] = module(self)
            if module_name not in self.module_ordering:
                self.module_ordering.append(module_name)
            return True

        finally:
            self.currently_loading.discard(module_name)