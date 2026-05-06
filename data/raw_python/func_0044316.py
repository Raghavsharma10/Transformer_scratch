def items(self):
        """An iterable of all (anchor-id, Anchor) mappings in the repository.
        """
        for anchor_id in self:
            try:
                anchor = self[anchor_id]
            except KeyError:
                assert False, 'Trying to load from missing file or something'

            yield (anchor_id, anchor)