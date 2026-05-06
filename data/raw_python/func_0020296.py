def uuid(self):
        '''Universally unique identifier for an instance of a :class:`Model`.
        '''
        pk = self.pkvalue()
        if not pk:
            raise self.DoesNotExist(
                'Object not saved. Cannot obtain universally unique id')
        return self.get_uuid(pk)