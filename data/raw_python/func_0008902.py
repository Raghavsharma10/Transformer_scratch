def fill_percent_done(self):
        """
        Calculate the percentage of edge pixels that would be done if the tile
        was reprocessed. This is done for each tile.
        """
        left = self.left
        right = self.right
        top = self.top
        bottom = self.bottom
        for i in xrange(self.n_chunks):
            self.percent_done.ravel()[i] = \
                np.sum([left.ravel()[i].percent_done,
                        right.ravel()[i].percent_done,
                        top.ravel()[i].percent_done,
                        bottom.ravel()[i].percent_done])
            self.percent_done.ravel()[i] /= \
                np.sum([left.ravel()[i].percent_done > 0,
                        right.ravel()[i].percent_done > 0,
                        top.ravel()[i].percent_done > 0,
                        bottom.ravel()[i].percent_done > 0, 1e-16])