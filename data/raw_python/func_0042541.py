def metas(self, prefix=None, limit=None, delimiter=None):
        """
        RETURN THE METADATA DESCRIPTORS FOR EACH KEY
        """
        limit = coalesce(limit, TOO_MANY_KEYS)
        keys = self.bucket.list(prefix=prefix, delimiter=delimiter)
        prefix_len = len(prefix)
        output = []
        for i, k in enumerate(k for k in keys if len(k.key) == prefix_len or k.key[prefix_len] in [".", ":"]):
            output.append({
                "key": strip_extension(k.key),
                "etag": convert.quote2string(k.etag),
                "expiry_date": Date(k.expiry_date),
                "last_modified": Date(k.last_modified)
            })
            if i >= limit:
                break
        return wrap(output)