def resolve_kw(self, kwargs):
        """ Resolve :kwargs: like `story_id: 1` to the form of `id: 1`.

        """
        resolved = {}
        for key, value in kwargs.items():
            split = key.split('_', 1)
            if len(split) > 1:
                key = split[1]
            resolved[key] = value
        return resolved