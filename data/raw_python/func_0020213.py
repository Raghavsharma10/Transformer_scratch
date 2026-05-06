def session(self, session=None):
        '''Override :meth:`Manager.session` so that this
        :class:`RelatedManager` can retrieve the session from the
        :attr:`related_instance` if available.
        '''
        if self.related_instance:
            session = self.related_instance.session
        # we have a session, we either create a new one return the same session
        if session is None:
            raise QuerySetError('Related manager can be accessed only from\
 a loaded instance of its related model.')
        return session