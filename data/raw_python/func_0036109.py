def remove(self, member):
        """Remove element from set; it must be a member.

        :raises KeyError: if the element is not a member.

        """
        if not self.client.srem(self.name, member):
            raise KeyError(member)