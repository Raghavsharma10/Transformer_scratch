def omero_bin(self, command):
        """
        Runs the omero command-line client with an array of arguments using the
        old environment
        """
        assert isinstance(command, list)
        if not self.old_env:
            raise Exception('Old environment not initialised')
        log.info("Running [old environment]: %s", " ".join(command))
        self.run('omero', command, capturestd=True, env=self.old_env)