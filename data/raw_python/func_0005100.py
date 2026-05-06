def read(self):
        # type: () -> Optional[str]
        """ Read the project version from .py file.

        This will regex search in the file for a
        ``__version__ = VERSION_STRING`` and read the version string.
        """
        with open(self.version_file) as fp:
            version = fp.read().strip()

            if is_valid(version):
                return version

            return None