def get_appstruct(self):
        """ return list of tuples keys and values corresponding to this model's
        data """
        result = []
        for k in self._get_keys():
            result.append((k, getattr(self, k)))
        return result