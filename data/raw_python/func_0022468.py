def index(self, *args, **kwargs):
        """Index documents, in the current session"""
        self.check_session()
        result = self.session.index(*args, **kwargs)
        if self.autosession:
            self.commit()
        return result