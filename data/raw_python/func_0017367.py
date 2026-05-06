def reorient_coordinates(self):
        """
        Returns a modified .verts array with new coordinates for nodes. 
        This does not need to modify .edges. The order of nodes, and therefore
        of verts rows is still the same because it is still based on the tree
        branching order (ladderized usually). 
        """
        # if tree is empty then bail out
        if len(self.ttree) < 2:
            return

        # down is the default orientation
        # down-facing tips align at y=0, first ladderized tip at x=0
        if self.ttree.style.orient in ('down', 0):
            pass

        # right-facing tips align at x=0, last ladderized tip at y=0
        elif self.ttree.style.orient in ('right', 3):

            # verts swap x and ys and make xs 0 to negative
            tmp = np.zeros(self.verts.shape)
            tmp[:, 1] = self.verts[:, 0]
            tmp[:, 0] = self.verts[:, 1] * -1
            self.verts = tmp

            # coords...
            tmp = np.zeros(self.coords.shape)
            tmp[:, 1] = self.coords[:, 0]
            tmp[:, 0] = self.coords[:, 1] * -1
            self.coords = tmp

        elif self.ttree.style.orient in ('left', 1):
            raise NotImplementedError("todo: left facing")

        else:
            raise NotImplementedError("todo: up facing")