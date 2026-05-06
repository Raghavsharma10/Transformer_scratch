def load_fields(self, *fields):
        '''Load extra fields to this :class:`StdModel`.'''
        if self._loadedfields is not None:
            if self.session is None:
                raise SessionNotAvailable('No session available')
            meta = self._meta
            kwargs = {meta.pkname(): self.pkvalue()}
            obj = session.query(self).load_only(fields).get(**kwargs)
            for name in fields:
                field = meta.dfields.get(name)
                if field is not None:
                    setattr(self, field.attname,
                            getattr(obj, field.attname, None))