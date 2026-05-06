def from_uuid(self, uuid, session=None):
        '''Retrieve a :class:`Model` from its universally unique identifier
``uuid``. If the ``uuid`` does not match any instance an exception will raise.
'''
        elems = uuid.split('.')
        if len(elems) == 2:
            model = get_model_from_hash(elems[0])
            if not model:
                raise Model.DoesNotExist(
                    'model id "{0}" not available'.format(elems[0]))
            if not session or session.router is not self:
                session = self.session()
            return session.query(model).get(id=elems[1])
        raise Model.DoesNotExist('uuid "{0}" not recognized'.format(uuid))