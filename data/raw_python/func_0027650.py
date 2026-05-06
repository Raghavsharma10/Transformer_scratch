def _lookup_node_parent(self, node):
        """
        Return the parent of the given node, based on an internal dictionary
        mapping of child nodes to the child's parent required since
        ElementTree doesn't make info about node ancestry/parentage available.
        """
        # Basic caching of our internal ancestry dict to help performance
        if not node in self.CACHED_ANCESTRY_DICT:
            # Given node isn't in cached ancestry dictionary, rebuild this now
            ancestry_dict = dict(
                (c, p) for p in self._impl_document.getiterator() for c in p)
            self.CACHED_ANCESTRY_DICT = ancestry_dict
        return self.CACHED_ANCESTRY_DICT[node]