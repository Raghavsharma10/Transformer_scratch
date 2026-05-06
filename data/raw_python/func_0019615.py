def is_installed(self):
        """
        Check if the tool is installed.
        
        Returns
        -------
        is_installed : bool
            True if the tool is installed.
        """
        return self.is_configured() and os.access(self.bin(), os.X_OK)