def israw(self):
        """
        Returns True if the PTY should operate in raw mode.

        If the container was not started with tty=True, this will return False.
        """

        if self.raw is None:
            info = self.container_info()
            self.raw = self.stdout.isatty() and info['Config']['Tty']

        return self.raw