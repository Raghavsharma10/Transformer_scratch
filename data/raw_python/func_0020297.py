def backend(self, client=None):
        '''The :class:`stdnet.BackendDatServer` for this instance.

        It can be ``None``.
        '''
        session = self.session
        if session:
            return session.model(self).backend