def extra(self, **kw):
        """
        Return a new S instance with extra args combined with existing
        set.
        """
        new = self._clone()
        actions = ['values_list', 'values_dict', 'order_by', 'query',
                   'filter', 'facet']
        for key, vals in kw.items():
            assert key in actions
            if hasattr(vals, 'items'):
                new.steps.append((key, vals.items()))
            else:
                new.steps.append((key, vals))
        return new