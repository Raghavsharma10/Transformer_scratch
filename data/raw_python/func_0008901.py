def fill_n_done(self):
        """
        Calculate and record the number of edge pixels that are done one each
        tile.
        """
        left = self.left
        right = self.right
        top = self.top
        bottom = self.bottom
        for i in xrange(self.n_chunks):
            self.n_done.ravel()[i] = np.sum([left.ravel()[i].n_done,
                                            right.ravel()[i].n_done,
                                            top.ravel()[i].n_done,
                                            bottom.ravel()[i].n_done])