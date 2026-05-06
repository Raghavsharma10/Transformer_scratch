def update_configurable(self, configurable_class, name, config):
        """
        Callback fired when a configurable instance is updated.

        Looks up the existing configurable in the proper "registry" and
        `apply_config()` is called on it.

        If a method named "on_<configurable classname>_update" is defined it
        is called in the work pool and passed the configurable's name, the old
        config and the new config.

        If the updated configurable is not present, `add_configurable()` is
        called instead.
        """
        configurable_class_name = configurable_class.__name__.lower()

        logger.info(
            "updating %s: '%s'", configurable_class_name, name
        )

        registry = self.registry_for(configurable_class)

        if name not in registry:
            logger.warn(
                "Tried to update unknown %s: '%s'",
                configurable_class_name, name
            )
            self.add_configurable(
                configurable_class,
                configurable_class.from_config(name, config)
            )
            return

        registry[name].apply_config(config)

        hook = self.hook_for(configurable_class, "update")
        if not hook:
            return

        def done(f):
            try:
                f.result()
            except Exception:
                logger.exception("Error updating configurable '%s'", name)

        self.work_pool.submit(hook, name, config).add_done_callback(done)