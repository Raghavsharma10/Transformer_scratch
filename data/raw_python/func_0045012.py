def quickload(self, *args):
        """
        Loads all manifests, loads all plugins, and then enables all plugins
        :param args: The args to pass to the plugin
        """
        self.load_manifests()
        self.load_plugins(args)
        self.enable_all_plugins()