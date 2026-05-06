def orb(self):
        '''The ORB used to access this object.

        This property's value will be None if no object above this object is a
        name server.

        '''
        with self._mutex:
            if self._parent.name == '/':
                return None
            return self._parent.orb