def bin(self, command):
        """
        Runs the omero command-line client with an array of arguments using the
        old environment
        """
        if isinstance(command, basestring):
            command = command.split()
        self.external.omero_bin(command)