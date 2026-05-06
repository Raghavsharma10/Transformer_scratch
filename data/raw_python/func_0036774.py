def remove_configurable(self, configurable_class, name):
        """
        Callback fired when a configurable instance is removed.

        Looks up the existing configurable in the proper "registry" and
        removes it.

        If a method named "on_<configurable classname>_remove" is defined it
        is called via the work pooland passed the configurable's name.

        If the removed configurable is not present, a warning is given and no
        further action is taken.
        """
        configurable_class_name = configurable_class.__name__.lower()

        logger.info("Removing %s: '%s'", configurable_class_name, name)

        registry = self.registry_for(configurable_class)

        if name not in registry:
            logger.warn(
                "Tried to remove unknown active %s: '%s'",
                configurable_class_name, name
            )
            return

        hook = self.hook_for(configurable_class, action="remove")
        if not hook:
            registry.pop(name)
            return

        def done(f):
            try:
                f.result()
                registry.pop(name)
            except Exception:
                logger.exception("Error removing configurable '%s'", name)

        self.work_pool.submit(hook, name).add_done_callback(done)