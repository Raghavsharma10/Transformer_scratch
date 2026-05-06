def is_running(self):
        """
        Check if the command is currently running

        Returns:
            bool: True if running, else False
        """
        if self.block:
            return False

        return self.thread.is_alive() or self.process.poll() is None