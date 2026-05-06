def ensure_directory_exists(self):
        """
        Make sure :attr:`directory` exists.

        :raises: :exc:`.MissingPasswordStoreError` when the password storage
                 directory doesn't exist.
        """
        if not os.path.isdir(self.directory):
            msg = "The password storage directory doesn't exist! (%s)"
            raise MissingPasswordStoreError(msg % self.directory)