def send_command(self, cmd):
        """Sends a generic command into FIFO.

        :param bytes cmd: Command chars to send into FIFO.

        """
        if not cmd:
            return

        with open(self.fifo, 'wb') as f:
            f.write(cmd)