def information_content(self):
        """Return the total information content of the motif.

        Return
        ------
        ic : float
            Motif information content.
        """
        ic = 0
        for row in self.pwm:
            ic += 2.0 + np.sum([row[x] * log(row[x])/log(2) for x in range(4) if row[x] > 0])
        return ic