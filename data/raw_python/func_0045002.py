def load_manifests(self):
        """
        Loads all plugin manifests on the plugin path
        """
        for path in self.plugin_paths:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    self.load_manifest(item_path)