def _dlog(self, msg, indent_increase=0):
        """log the message to the log"""
        self._log.debug("interp", msg, indent_increase, filename=self._orig_filename, coord=self._coord)