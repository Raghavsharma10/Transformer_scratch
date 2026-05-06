def _check_rest_version(self, version):
        """Validate a REST API version is supported by the library and target array."""
        version = str(version)

        if version not in self.supported_rest_versions:
            msg = "Library is incompatible with REST API version {0}"
            raise ValueError(msg.format(version))

        array_rest_versions = self._list_available_rest_versions()
        if version not in array_rest_versions:
            msg = "Array is incompatible with REST API version {0}"
            raise ValueError(msg.format(version))

        return LooseVersion(version)