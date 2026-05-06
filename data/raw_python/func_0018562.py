def omero_cli(self, command):
        """
        Runs a command as if from the OMERO command-line without the need
        for using popen or subprocess.
        """
        assert isinstance(command, list)
        if not self.cli:
            raise Exception('omero.cli not initialised')
        log.info("Invoking CLI [current environment]: %s", " ".join(command))
        self.cli.invoke(command, strict=True)