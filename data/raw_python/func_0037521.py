def visible(self, status):
        """Turn the display on/off (quickly)"""
        self._display_control = ByteUtil.apply_flag(self._display_control, Command.DISPLAY_ON, status)
        self.command(self._display_control)