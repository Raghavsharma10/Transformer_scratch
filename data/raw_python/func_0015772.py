def facet_raw(self, **kw):
        """
        Return a new S instance with raw facet args combined with
        existing set.
        """
        items = kw.items()
        if six.PY3:
            items = list(items)
        return self._clone(next_step=('facet_raw', items))