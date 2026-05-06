def _router_numbers(self):
        """A tuple of the numbers of all "routing" basins."""
        return tuple(up for up in self._up2down.keys()
                     if up in self._up2down.values())