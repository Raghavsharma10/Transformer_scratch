def cursor(self, status):
        """Turn underline cursor visibility on/off"""
        self._display_control = ByteUtil.apply_flag(self._display_control, Command.CURSOR_ON, status)
        self.command(self._display_control)