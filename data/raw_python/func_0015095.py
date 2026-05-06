def _encode_utf8(self, **kwargs):
        """
        UTF8 encodes all of the NVP values.
        """
        if is_py3:
            # This is only valid for Python 2. In Python 3, unicode is
            # everywhere (yay).
            return kwargs

        unencoded_pairs = kwargs
        for i in unencoded_pairs.keys():
            #noinspection PyUnresolvedReferences
            if isinstance(unencoded_pairs[i], types.UnicodeType):
                unencoded_pairs[i] = unencoded_pairs[i].encode('utf-8')
        return unencoded_pairs