def onex(self):
        """
        delete all X columns except the first one.
        """
        xCols=[i for i in range(self.nCols) if self.colTypes[i]==3]
        if len(xCols)>1:
            for colI in xCols[1:][::-1]:
                self.colDelete(colI)