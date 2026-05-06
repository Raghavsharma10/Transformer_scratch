def alignXY(self):
        """aligns XY pairs (or XYYY etc) by X value."""

        # figure out what data we have and will align to
        xVals=[]
        xCols=[x for x in range(self.nCols) if self.colTypes[x]==3]
        yCols=[x for x in range(self.nCols) if self.colTypes[x]==0]
        xCols,yCols=np.array(xCols),np.array(yCols)
        for xCol in xCols:
            xVals.extend(self.colData[xCol])
        #xVals=list(np.round(set(xVals),5))
        xVals=list(sorted(list(set(xVals))))

        # prepare our new aligned dataset
        newData=np.empty(len(xVals)*self.nCols)
        newData[:]=np.nan
        newData=newData.reshape(len(xVals),self.nCols)
        oldData=np.round(self.data,5)

        # do the alignment
        for xCol in xCols:
            columnsToShift=[xCol]
            for col in range(xCol+1,self.nCols):
                if self.colTypes[col]==0:
                    columnsToShift.append(col)
                else:
                    break
            # determine how to move each row
            for row in range(len(oldData)):
                oldXvalue=oldData[row,xCol]
                if oldXvalue in xVals:
                    newRow=xVals.index(oldXvalue)
                    newData[newRow,columnsToShift]=oldData[row,columnsToShift]

        # commit changes
        newData[:,0]=xVals
        self.data=newData
        self.onex()