def query(self, session=None):
        '''Returns a new :class:`Query` for :attr:`Manager.model`.'''
        if session is None or session.router is not self.router:
            session = self.session()
        return session.query(self.model)