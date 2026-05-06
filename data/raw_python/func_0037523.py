def blink(self, status):
        """Turn blink cursor visibility on/off"""
        self._display_control = ByteUtil.apply_flag(self._display_control, Command.BLINKING_ON, status)
        self.command(self._display_control)