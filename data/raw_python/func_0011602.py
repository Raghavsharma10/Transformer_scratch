def get_queryset(self):
        """Get the queryset for the action.

        If action is read action, return a CachedQueryset
        Otherwise, return a Django queryset
        """
        queryset = super(CachedViewMixin, self).get_queryset()
        if self.action in ('list', 'retrieve'):
            return CachedQueryset(self.get_queryset_cache(), queryset=queryset)
        else:
            return queryset