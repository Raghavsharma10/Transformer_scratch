def bin(self):
        """
        Get the command used to run the tool.

        Returns
        -------
        command : str
            The tool system command.
        """
        if self.local_bin:
            return self.local_bin
        else:
            return self.config.bin(self.name)