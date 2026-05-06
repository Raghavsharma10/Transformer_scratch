def get_topology_id(self, attr="name"):
        """
        Returns the unique ID representing the topology of the current tree. 
        Two trees with the same topology will produce the same id. If trees are
        unrooted, make sure that the root node is not binary or use the
        tree.unroot() function before generating the topology id.

        This is useful to detect the number of unique topologies over a bunch 
        of trees, without requiring full distance methods.

        The id is, by default, calculated based on the terminal node's names. 
        Any other node attribute could be used instead.
        """
        edge_keys = []
        for s1, s2 in self.get_edges():
            k1 = sorted([getattr(e, attr) for e in s1])
            k2 = sorted([getattr(e, attr) for e in s2])
            edge_keys.append(sorted([k1, k2]))
        return md5(str(sorted(edge_keys)).encode('utf-8')).hexdigest()