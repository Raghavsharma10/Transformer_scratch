def structure(self, instance, client=None):
        '''Create a backend :class:`stdnet.odm.Structure` handler.

        :param instance: a :class:`stdnet.odm.Structure`
        :param client: Optional client handler.
        '''
        struct = self.struct_map.get(instance._meta.name)
        if struct is None:
            raise ModelNotAvailable('"%s" is not available for backend '
                                    '"%s"' % (instance._meta.name, self))
        client = client if client is not None else self.client
        return struct(instance, self, client)