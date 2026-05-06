def resize(self, size=None):
        """
        Resize the container's PTY.

        If `size` is not None, it must be a tuple of (height,width), otherwise
        it will be determined by the size of the current TTY.
        """

        if not self.israw():
            return

        size = size or tty.size(self.stdout)

        if size is not None:
            rows, cols = size
            try:
                self.client.resize(self.container, height=rows, width=cols)
            except IOError: # Container already exited
                pass