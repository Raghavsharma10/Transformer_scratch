def _get_whole_subtrees(self):
        """Returns an array of nodes in the tree that have balanced subtrees beneath them,
        moving from left to right.
        """
        subtrees = []
        loose_leaves = len(self.leaves) - 2**int(log(len(self.leaves), 2))
        the_node = self.root
        while loose_leaves:
            subtrees.append(the_node.l)
            the_node = the_node.r
            loose_leaves = loose_leaves - 2**int(log(loose_leaves, 2))
        subtrees.append(the_node)
        return subtrees