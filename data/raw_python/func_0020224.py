def backend(self):
        '''Returns the :class:`stdnet.BackendStructure`.
        '''
        session = self.session
        if session is not None:
            if self._field:
                return session.model(self._field.model).backend
            else:
                return session.model(self).backend