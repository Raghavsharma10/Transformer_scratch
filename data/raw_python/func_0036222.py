def set_datarate(self, rate, rfm=1):
        """Set datarate (baudrate)."""
        cmds = {1: 'r', 2: 'R'}
        self._write_cmd('{}{}'.format(rate, cmds[rfm]))