def _pushMessages(self):
        """ Internal callback used to make sure the msg list keeps moving. """
        # This continues to get itself called until no msgs are left in list.
        self.showStatus('')
        if len(self._statusMsgsToShow) > 0:
            self.top.after(200, self._pushMessages)