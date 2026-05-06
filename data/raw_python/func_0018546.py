def run(self, command):
        """
        Runs a command as if from the command-line
        without the need for using popen or subprocess
        """
        if isinstance(command, basestring):
            command = command.split()
        else:
            command = list(command)
        self.external.omero_cli(command)