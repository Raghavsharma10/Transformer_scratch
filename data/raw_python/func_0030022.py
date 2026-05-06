def dict(self):
        """A dict that holds key/values for all of the properties in the
        object.

        :return:

        """
        d = {p.key: getattr(self, p.key) for p in self.__mapper__.attrs
             if p.key not in ('table', 'stats', '_codes', 'data')}

        if not d:
            raise Exception(self.__dict__)

        d['schema_type'] = self.schema_type

        if self.data:
            # Copy data fields into top level dict, but don't overwrite existind values.
            for k, v in six.iteritems(self.data):
                if k not in d and k not in ('table', 'stats', '_codes', 'data'):
                    d[k] = v

        return d