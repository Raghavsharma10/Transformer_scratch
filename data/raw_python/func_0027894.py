def data(self):
        """return all of colData as a 2D numpy array."""
        data=np.empty((self.nRows,self.nCols),dtype=np.float)
        data[:]=np.nan # make everything nan by default
        for colNum,colData in enumerate(self.colData):
            validIs=np.where([np.isreal(v) for v in colData])[0]
            validData=np.ones(len(colData))*np.nan
            validData[validIs]=np.array(colData)[validIs]
            data[:len(colData),colNum]=validData # only fill cells that have data

        return data