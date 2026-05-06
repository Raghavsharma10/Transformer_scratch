def get_event_kind(self):
        """
        Unless we're on the front page we'll have a kind_slug like 'movies'.
        We need to translate that into an event `kind` like 'movie'.
        """
        slug = self.kwargs.get('kind_slug', None)
        if slug is None:
            return None  # Front page; showing all Event kinds.
        else:
            slugs_to_kinds = {v:k for k,v in Event.KIND_SLUGS.items()}
            return slugs_to_kinds.get(slug, None)