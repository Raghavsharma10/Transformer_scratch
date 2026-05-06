def iter_causes(self):
        """Iterate over all causes."""
        curr = self._cause
        while curr is not None:
            yield curr
            curr = curr._cause