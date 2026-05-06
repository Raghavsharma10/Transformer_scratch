def read_backend(self, client=None):
        '''The read :class:`stdnet.BackendDatServer` for this instance.

        It can be ``None``.
        '''
        session = self.session
        if session:
            return session.model(self).read_backend