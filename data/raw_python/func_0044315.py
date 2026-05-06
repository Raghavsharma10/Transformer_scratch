def add(self, anchor):
        """Add a new anchor to the repository.

        This will create a new ID for the anchor and provision new storage for
        it.

        Returns: The storage ID for the Anchor which can be used to retrieve
            the anchor later.

        """
        anchor_id = uuid.uuid4().hex
        anchor_path = self._anchor_path(anchor_id)
        with anchor_path.open(mode='wt') as f:
            save_anchor(f, anchor, self.root)

        return anchor_id