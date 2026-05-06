def add_adjust(self, data, prehashed=False):
        """Add a new leaf, and adjust the tree, without rebuilding the whole thing.
        """
        subtrees = self._get_whole_subtrees()
        new_node = Node(data, prehashed=prehashed)
        self.leaves.append(new_node)
        for node in reversed(subtrees):
            new_parent = Node(node.val + new_node.val)
            node.p, new_node.p = new_parent, new_parent
            new_parent.l, new_parent.r = node, new_node
            node.sib, new_node.sib = new_node, node
            node.side, new_node.side = 'L', 'R'
            new_node = new_node.p
        self.root = new_node