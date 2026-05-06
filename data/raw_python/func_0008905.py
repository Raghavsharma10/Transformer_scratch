def find_best_candidate(self):
        """
        Determine which tile, when processed, would complete the largest
        percentage of unresolved edge pixels. This is a heuristic function
        and does not give the optimal tile.
        """
        self.fill_percent_done()
        i_b = np.argmax(self.percent_done.ravel())
        if self.percent_done.ravel()[i_b] <= 0:
            return None

        # check for ties
        I = self.percent_done.ravel() == self.percent_done.ravel()[i_b]
        if I.sum() == 1:
            return i_b
        else:
            I2 = np.argmax(self.max_elev.ravel()[I])
            return I.nonzero()[0][I2]