def tags(self):
        # type: () -> Set[str]
        """
        Tags applied to operation.
        """
        tags = set()
        if self._tags:
            tags.update(self._tags)
        if self.binding:
            binding_tags = getattr(self.binding, 'tags', None)
            if binding_tags:
                tags.update(binding_tags)
        return tags