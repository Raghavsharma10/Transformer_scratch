def get_distance(self, target, target2=None, topology_only=False):
        """
        Returns the distance between two nodes. If only one target is
        specified, it returns the distance bewtween the target and the
        current node.
    
        Parameters:
        -----------
        target: 
            a node within the same tree structure.

        target2: 
            a node within the same tree structure. If not specified, 
            current node is used as target2.

        topology_only: 
            If set to True, distance will refer to the number of nodes 
            between target and target2.

        Returns:
        --------
        branch length distance between target and target2. If topology_only 
        flag is True, returns the number of nodes between target and target2.
        """
        if target2 is None:
            target2 = self
            root = self.get_tree_root()
        else:
            # is target node under current node?
            root = self

        target, target2 = _translate_nodes(root, target, target2)
        ancestor = root.get_common_ancestor(target, target2)

        dist = 0.0
        for n in [target2, target]:
            current = n
            while current != ancestor:
                if topology_only:
                    if  current!=target:
                        dist += 1
                else:
                    dist += current.dist
                current = current.up
        return dist