def pop(self):
        """Remove and return an arbitrary set element.

        :raises KeyError: if the set is empty.

        """
        member = self.client.spop(self.name)
        if member is not None:
            return member
        raise KeyError()