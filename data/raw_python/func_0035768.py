def _create_structure(self):
        """Create necessary structure to hold a dataset."""

        # Ensure that the specified path does not exist and create it.
        if os.path.exists(self._abspath):
            raise(StorageBrokerOSError(
                "Path already exists: {}".format(self._abspath)
            ))

        # Make sure the parent directory exists.
        parent, _ = os.path.split(self._abspath)
        if not os.path.isdir(parent):
            raise(StorageBrokerOSError(
                "No such directory: {}".format(parent)))

        os.mkdir(self._abspath)

        # Create more essential subdirectories.
        for abspath in self._essential_subdirectories:
            if not os.path.isdir(abspath):
                os.mkdir(abspath)