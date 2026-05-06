def optimize(self):
        """Optimize index for faster by-document-id queries."""
        self.check_session()
        result = self.session.optimize()
        if self.autosession:
            self.commit()
        return result