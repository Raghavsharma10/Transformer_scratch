def get_list_url(self, kind_slug=None):
        """
        Get the list URL for this Work.
        You can also pass a kind_slug in (e.g. 'movies') and it will use that
        instead of the Work's kind_slug. (Why? Useful in views. Or tests of
        views, at least.)
        """
        if kind_slug is None:
            kind_slug = self.KIND_SLUGS[self.kind]
        return reverse('spectator:events:work_list',
                                            kwargs={'kind_slug': kind_slug})