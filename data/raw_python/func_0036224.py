def set_toggle_mask(self, mode_mask, rfm=1):
        """Set toggle baudrate mask.

        The baudrate mask values are:
          1: 17.241 kbps
          2 : 9.579 kbps
          4 : 8.842 kbps
        These values can be or'ed.
        """
        cmds = {1: 'm', 2: 'M'}
        self._write_cmd('{}{}'.format(mode_mask, cmds[rfm]))