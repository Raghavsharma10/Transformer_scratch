def ladderize(self, direction=0):
        """
        Ladderize tree (order descendants) so that top child has fewer 
        descendants than the bottom child in a left to right tree plot. 
        To reverse this pattern use direction=1.
        """
        nself = deepcopy(self)
        nself.treenode.ladderize(direction=direction)
        nself._fixed_order = None
        nself._coords.update()
        return nself