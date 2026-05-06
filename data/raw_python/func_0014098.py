def _save_to_checkpoint(self, filename):
        """Save the current running config to the given file."""
        command = 'checkpoint file {}'.format(filename)
        self.device.send_command(command)