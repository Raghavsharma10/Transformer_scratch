def pcc_pos(self, row1, row2):
        """Calculate the Pearson correlation coefficient of one position
        compared to another position.

        Returns
        -------
        score : float
            Pearson correlation coefficient.
        """
        mean1 = np.mean(row1)
        mean2 = np.mean(row2)

        a = 0
        x = 0
        y = 0
        for n1, n2 in zip(row1, row2):
            a += (n1 - mean1) * (n2 - mean2)
            x += (n1 - mean1) ** 2
            y += (n2 - mean2) ** 2
        
        if a == 0:
            return 0
        else:
            return a / sqrt(x * y)