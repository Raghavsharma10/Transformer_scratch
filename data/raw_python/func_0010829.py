def derive_ordering(self):
        """
        Returns what field should be used for ordering (using a prepended '-' to indicate descending sort).

        If the default order of the queryset should be used, returns None
        """
        if '_order' in self.request.GET:
            return self.request.GET['_order']
        elif self.default_order:
            return self.default_order
        else:
            return None