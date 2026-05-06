def _get_stitch_report_path(self):
        """Checks if stitch report label / path is set. Returns absolute path."""
        if self.Parameters['-r'].isOn():
            stitch_path = self._absolute(str(self.Parameters['-r'].Value))
            return stitch_path
        elif self.Parameters['-r'].isOff():
            return None