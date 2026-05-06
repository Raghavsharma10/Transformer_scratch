def get_context_data(self, **kwargs):
        """
        Returns view context dictionary.

        :rtype: dict.
        """
        kwargs.update({
            'entries': Entry.objects.get_for_tag(
                self.kwargs.get('slug', 0)
            )
        })

        return super(EntriesView, self).get_context_data(**kwargs)