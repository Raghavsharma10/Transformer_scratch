def load_plugins(self, *args):
        """
        Loads all plugins

        :param args: Arguments to pass to the plugins
        """
        for manifest in self._manifests:
            self.load_plugin(manifest, *args)