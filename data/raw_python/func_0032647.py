def create(cls, **kw):
        """
        Create an instance of this class, first cleaning up the keyword
        arguments so they will fill in any required values.

        @return: an instance of C{cls}
        """
        for k, v in kw.items():
            attr = getattr(cls, k, None)
            if isinstance(attr, RecordAttribute):
                kw.pop(k)
                kw.update(attr._decompose(v))
        return cls(**kw)