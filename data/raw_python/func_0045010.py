def reload_all_plugins(self, *args):
        """
        Reloads all initialized plugins
        """
        for manifest in self._manifests[:]:
            if self.get_plugin(manifest["name"]) is not None:
                self.reload_plugin(manifest["name"], *args)