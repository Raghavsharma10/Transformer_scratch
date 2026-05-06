def iter_edges(self, cached_content=None):
        """
        Iterate over the list of edges of a tree. Each egde is represented as a
        tuple of two elements, each containing the list of nodes separated by
        the edge.
        """
        if not cached_content:
            cached_content = self.get_cached_content()
        all_leaves = cached_content[self]
        for n, side1 in six.iteritems(cached_content):
            yield (side1, all_leaves - side1)