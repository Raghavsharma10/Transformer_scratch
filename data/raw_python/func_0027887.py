def colDelete(self,colI=-1):
        """delete a column at a single index. Negative numbers count from the end."""
#        print("DELETING COLUMN: [%d] %s"%(colI,self.colDesc[colI]))
        self.colNames.pop(colI)
        self.colDesc.pop(colI)
        self.colUnits.pop(colI)
        self.colComments.pop(colI)
        self.colTypes.pop(colI)
        self.colData.pop(colI)
        return