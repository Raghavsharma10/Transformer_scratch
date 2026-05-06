def search_nodes(self, **conditions):
        """
        Returns the list of nodes matching a given set of conditions.
        **Example:**
        tree.search_nodes(dist=0.0, name="human")
        """
        matching_nodes = []
        for n in self.iter_search_nodes(**conditions):
            matching_nodes.append(n)
        return matching_nodes