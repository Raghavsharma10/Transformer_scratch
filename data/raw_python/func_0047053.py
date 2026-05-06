def reload_modules(self):
        """(Re)load all of the configured modules.

        1. Calls stop(reloading=True) on each loaded module
        2. Clears .loaded_modules and .module_ordering
        3. Loads each module specified in the config
        4. Calls start() on each loaded module, with reloading set depending
           on whether the module was previously loaded or not
        5. Dispatches the STARTUP event, since all modules have been rebooted

        Returns True if all modules reloaded successfully, otherwise False.
        """
        old_modules = set(self.loaded_modules)
        for module in self.loaded_modules.itervalues():
            module.stop(reloading=True)

        self.loaded_modules = {}
        self.module_ordering = []

        try:
            modules_to_load = sorted(self.config.items("modules"),
                                     key=lambda x:int(x[1]))
        except (TypeError,ValueError):
            _log.exception("Unable to load modules due to invalid priority.")
            return False

        modules_success = []
        modules_failure = []

        for module_name,_ in modules_to_load:
            if self.load_module(module_name):
                modules_success.append(module_name)
            else:
                modules_failure.append(module_name)

        if modules_success:
            _log.info("Loaded the following modules: %s", modules_success)
        if modules_failure:
            _log.error("These modules failed to load: %s", modules_failure)

        for module_name in self.module_ordering:
            module = self.loaded_modules[module_name]
            module.start(reloading=(module_name in old_modules))

        self.process_event("STARTUP", self.client, (), force_dispatch=True)

        return not modules_failure