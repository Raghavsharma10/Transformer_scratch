def reload_all_manifests(self):
        """
        Reloads all loaded manifests, and loads any new manifests
        """
        self._logger.debug("Reloading all manifests.")
        self._manifests = []
        self.load_manifests()
        self._logger.debug("All manifests reloaded.")