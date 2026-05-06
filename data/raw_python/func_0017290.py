def get_edge_values_from_dict(self, node_value_dict=None, include_stem=True):
        """
        Enter a dictionary mapping node 'idx' or tuple of tipnames to values 
        that you want mapped to the stem and descendant edges that node. 
        Edge values are returned in proper plot order to be entered to the 
        edge_colors or edge_widths arguments to draw(). To see node idx values 
        use node_labels=True in draw(). If dictionary keys are integers it is
        assumed they are node idxs. 

        Note: it is safer to use tip labels to identify clades than node idxs 
        since tree tranformations (e.g., rooting) can change the mapping of 
        idx values to nodes on the tree.

        This function is most convenient for applying values to clades. To
        instead map values to specific edges (e.g., a single internal edge) 
        it will be easier to use tre.get_edge_values() and then to set the 
        values of the internal edges manually.

        Example 1: 
          tre = toytree.tree("((a,b),(c,d));")
          tre.get_edge_values_from_dict({5: 'green', 6: 'red'})
          # ['green', 'green', 'green', 'red', 'red', 'red']

        Example 2: 
          tre = toytree.tree("((a,b),(c,d));")
          tre.get_edge_values_from_dict({(a, b): 'green', (c, d): 'red'})          
          # ['green', 'green', 'green', 'red', 'red', 'red']
        """
        # map node idxs to the order in which edges are plotted
        idxs = {j: i for (i, j) in enumerate(self.get_edge_values())}
        values = [None] * self._coords.edges.shape[0]
        if node_value_dict is None:
            return values

        # convert tipname lists to node idxs
        rmap = {}
        for (key, val) in node_value_dict.items():
            if isinstance(key, (str, tuple)):
                node = fuzzy_match_tipnames(self, key, None, None, True, False)
                rmap[node.idx] = val
            else:
                rmap[key] = val
        node_value_dict = rmap

        # map over tree
        for node in self.treenode.traverse("levelorder"):
            if node.idx in node_value_dict:

                # add value to stem edge
                if include_stem:
                    if not node.is_root():
                        values[idxs[node.idx]] = node_value_dict[node.idx]
            
                # add value to descendants edges
                for desc in node.get_descendants():
                    values[idxs[desc.idx]] = node_value_dict[node.idx]
        return values