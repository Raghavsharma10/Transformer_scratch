def _build(self, leaves):
        """Private helper function to create the next aggregation level and put all references in place.
        """
        new, odd = [], None
        # check if even number of leaves, promote odd leaf to next level, if not
        if len(leaves) % 2 == 1:
            odd = leaves.pop(-1)
        for i in range(0, len(leaves), 2):
            newnode = Node(leaves[i].val + leaves[i + 1].val)
            newnode.l, newnode.r = leaves[i], leaves[i + 1]
            leaves[i].side, leaves[i + 1].side, leaves[i].p, leaves[i + 1].p = 'L', 'R', newnode, newnode
            leaves[i].sib, leaves[i + 1].sib = leaves[i + 1], leaves[i]
            new.append(newnode)
        if odd:
            new.append(odd)
        return new