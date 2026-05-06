def nRows(self):
        """returns maximum number of rows based on the longest colData"""
        if self.nCols: return max([len(x) for x in self.colData])
        else: return 0