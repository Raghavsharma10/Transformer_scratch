def _encode_key(self, obj):
        """Encodes a dictionary key - a key can only be a string in std JSON"""

        if obj.__class__ is str:
            return self._encode_str(obj)

        if obj.__class__ is UUID:
            return '"' + str(obj) + '"'

        # __mm_serialize__ is called before any isinstance checks (but after exact type checks)
        try:
            sx_encoder = obj.__mm_serialize__
        except AttributeError:
            pass
        else:
            try:
                data = sx_encoder()
            except NotImplementedError:
                pass
            else:
                return self._encode_key(data)

        if isinstance(obj, UUID):
            return '"' + str(obj) + '"'

        if isinstance(obj, str):
            return self._encode_str(obj)

        # if everything else failed try the default() method and re-raise any TypeError
        # exceptions as more specific "not a valid dict key" TypeErrors
        try:
            value = self.default(obj)
        except TypeError:
            raise TypeError('{!r} is not a valid dictionary key'.format(obj))

        return self._encode_key(value)