def set_frequency(self, frequency, rfm=1):
        """Set frequency in kHz.

        The frequency can be set in 5kHz steps.
        """
        cmds = {1: 'f', 2: 'F'}
        self._write_cmd('{}{}'.format(frequency, cmds[rfm]))