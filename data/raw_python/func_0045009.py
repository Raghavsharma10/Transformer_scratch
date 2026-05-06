def reload_plugin(self, name, *args):
        """
        Reloads a given plugin

        :param name: The name of the plugin
        :param args: The args to pass to the plugin
        """
        self._logger.debug("Reloading {}.".format(name))

        self._logger.debug("Disabling {}.".format(name))
        self.get_plugin(name).disable()

        self._logger.debug("Removing plugin instance.")
        del self._plugins[name]

        self._logger.debug("Unloading module.")
        del self._modules[name]

        self._logger.debug("Reloading manifest.")
        old_manifest = self.get_manifest(name)
        self._manifests.remove(old_manifest)
        self.load_manifest(old_manifest["path"])

        self._logger.debug("Loading {}.".format(name))
        self.load_plugin(self.get_manifest(name), *args)

        self._logger.debug("Enabling {}.".format(name))
        self.get_plugin(name).enable()

        self._logger.debug("Plugin {} reloaded.".format(name))