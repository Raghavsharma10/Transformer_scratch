def add(self, value, session=None, **kwargs):
        '''Add ``value``, an instance of :attr:`formodel` to the
:attr:`through` model. This method can only be accessed by an instance of the
model for which this related manager is an attribute.'''
        s, instance = self.session_instance('add', value, session, **kwargs)
        return s.add(instance)