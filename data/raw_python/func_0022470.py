def delete(self, docids):
        """Delete documents from the current session."""
        self.check_session()
        result = self.session.delete(docids)
        if self.autosession:
            self.commit()
        return result