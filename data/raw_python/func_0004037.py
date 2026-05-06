def has_descriptor(self, descriptor):
        """
        Return ``True`` if the character has the given descriptor.

        :param IPADescriptor descriptor: the descriptor to be checked against
        :rtype: bool
        """
        for p in self.descriptors:
            if p in descriptor:
                return True
        return False