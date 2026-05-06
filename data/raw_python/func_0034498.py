def install_key(self, keyfile, authorized_keys):
        """Install a key into the authorized_keys file."""

        # Make the directory containing the authorized_keys
        # file, if it doesn't exist. (Typically ~/.ssh).
        # Ignore errors; we'll fail shortly if we can't
        # create the authkeys file.
        try:
            os.makedirs(os.path.dirname(authorized_keys), 0o700)
        except OSError:
            pass

        keydata = open(keyfile).read()
        target_fd = os.open(authorized_keys, os.O_RDWR | os.O_CREAT, 0o600)
        self.install_key_data(keydata, os.fdopen(target_fd, 'w+'))