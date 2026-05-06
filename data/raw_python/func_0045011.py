def unload_plugin(self, name):
        """
        Unloads a specified plugin
        :param name: The name of the plugin
        """
        self._logger.debug("Unloading {}.".format(name))

        self._logger.debug("Removing plugin instance.")
        del self._plugins[name]

        self._logger.debug("Unloading module.")
        del self._modules[name]

        self._logger.debug("Unloading manifest...")
        manifest = self.get_manifest(name)
        self._manifests.remove(manifest)

        self._logger.debug("{} unloaded.".format(name))