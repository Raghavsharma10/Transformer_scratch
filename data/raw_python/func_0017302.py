def unroot(self):
        """
        Returns a copy of the tree unrooted. Does not transform tree in-place.
        """
        nself = self.copy()
        nself.treenode.unroot()
        nself.treenode.ladderize()
        nself._coords.update()
        return nself