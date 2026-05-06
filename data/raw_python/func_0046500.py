def _decode_meta(self, meta, **extra):
        """
        Decode and load underlying meta structure to dict and apply optional extra values.
        """
        _meta = json.loads(meta) if meta else {}
        _meta.update(extra)
        return _meta