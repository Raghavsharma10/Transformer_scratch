def set_toggle_interval(self, interval, rfm=1):
        """Set the toggle interval."""
        cmds = {1: 't', 2: 'T'}
        self._write_cmd('{}{}'.format(interval, cmds[rfm]))