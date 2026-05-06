def get_ordering(self):
        """
        Returns ordering value for list.

        :rtype: str.
        """
        #noinspection PyUnresolvedReferences
        ordering = self.request.GET.get('ordering', None)

        if ordering not in ['title', '-created_at']:
            ordering = '-created_at'

        return ordering