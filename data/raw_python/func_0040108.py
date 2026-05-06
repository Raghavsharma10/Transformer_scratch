def _validate_extension(self):
        """Validates that source file extension is supported.

        :raises: UnsupportedExtensionError

        """
        extension = self.fpath.split('.')[-1]
        if extension not in self.supported_extensions:
            raise UnsupportedExtensionError