def get_queryset(self, **kwargs):
        """
        Gets our queryset.  This takes care of filtering if there are any
        fields to filter by.
        """
        queryset = self.derive_queryset(**kwargs)

        return self.order_queryset(queryset)