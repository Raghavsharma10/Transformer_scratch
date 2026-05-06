def _choose_rest_version(self):
        """Return the newest REST API version supported by target array."""
        versions = self._list_available_rest_versions()
        versions = [LooseVersion(x) for x in versions if x in self.supported_rest_versions]
        if versions:
            return max(versions)
        else:
            raise PureError(
                "Library is incompatible with all REST API versions supported"
                "by the target array.")