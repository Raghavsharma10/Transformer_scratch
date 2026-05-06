def reload_manifest(self, manifest):
        """
        Reloads a manifest from the disk
        :param manifest: The manifest to reload
        """
        self._logger.debug("Reloading manifest for {}.".format(manifest.get("name", "Unnamed Plugin")))
        self._manifests.remove(manifest)
        self.load_manifest(manifest["path"])
        self._logger.debug("Manifest reloaded.")