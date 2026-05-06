def _merge_meta(self, encoded_meta, meta):
        """
        Merge new meta dict into encoded meta. Returns new encoded meta.
        """
        new_meta = None

        if meta:
            _meta = self._decode_meta(encoded_meta)
            for key, value in six.iteritems(meta):
                if value is None:
                    _meta.pop(key, None)
                else:
                    _meta[key] = value
            new_meta = self._encode_meta(_meta)

        return new_meta