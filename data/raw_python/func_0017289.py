def get_edge_values(self, feature='idx'):
        """
        Returns edge values in the order they are plotted (see .get_edges())
        """
        elist = []
        for cidx in self._coords.edges[:, 1]:
            node = self.treenode.search_nodes(idx=cidx)[0]
            elist.append(
                (node.__getattribute__(feature) if hasattr(node, feature) else "")
                )
        return elist