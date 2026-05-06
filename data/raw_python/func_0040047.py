def serialize(self):
        """
        Converts :class:`adnpy.models.Model` into a normal dict without references to the api
        """

        data = {}
        for k, v in self.iteritems():
            if k.startswith('_'):
                continue

            if isinstance(v, APIModel):
                data[k] = v.serialize()
            elif v and is_seq_not_string(v) and isinstance(v[0], APIModel):
                data[k] = [x.serialize() for x in v]
            else:
                data[k] = v

        return data