def ic_pos(self, row1, row2=None):
        """Calculate the information content of one position.

        Returns
        -------
        score : float
            Information content.
        """
        if row2 is None:
            row2 = [0.25,0.25,0.25,0.25]

        score = 0
        for a,b in zip(row1, row2):
            if a > 0:
                score += a * log(a / b) / log(2)
        return score