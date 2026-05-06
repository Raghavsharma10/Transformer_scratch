def drop_index(self, keep_model=True):
        """Drop all indexed documents from the session. Optionally, drop model too."""
        self.check_session()
        result = self.session.drop_index(keep_model)
        if self.autosession:
            self.commit()
        return result