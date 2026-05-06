def canonical_value(self, query):
        """
        Return the canonical value corresponding to the given query value.

        Return ``None`` if the query value is not present in any descriptor of the group.

        :param str query: the descriptor value to be checked against
        """
        for d in self.descriptors:
            if query in d:
                return d.canonical_label
        return None