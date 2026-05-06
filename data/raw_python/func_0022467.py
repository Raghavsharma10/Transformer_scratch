def buffer(self, *args, **kwargs):
        """Buffer documents, in the current session"""
        self.check_session()
        result = self.session.buffer(*args, **kwargs)
        return result