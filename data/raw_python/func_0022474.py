def set_autosession(self, value=None):
        """
        Turn autosession (automatic committing after each modification call) on/off.
        If value is None, only query the current value (don't change anything).
        """
        if value is not None:
            self.rollback()
            self.autosession = value
        return self.autosession