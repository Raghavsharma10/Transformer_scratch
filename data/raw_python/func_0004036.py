def dg_value(self, descriptor_group):
        """
        Return the canonical value of a descriptor of the character,
        provided it is present in the given descriptor group.

        If not present, return ``None``.

        :param IPADescriptorGroup descriptor_group: the descriptor group to be checked against
        :rtype: str
        """
        for p in self.descriptors:
            if p in descriptor_group:
                return descriptor_group.canonical_value(p)
        return None