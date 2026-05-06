def execute(self, command):
        """Primary method to execute ipmitool commands
        :param command: ipmi command to execute, str or list
        
        e.g.
        > ipmi = ipmitool('consolename.prod', 'secretpass')
        > ipmi.execute('chassis status')
        >
        """
        if isinstance(command, str):
            self.method(command.split())
        elif isinstance(command, list):
            self.method(command)
        else:
            raise TypeError("command should be either a string or list type")

        if self.error:
            raise IPMIError(self.error)
        else:
            return self.status