def data(self,data):
        """Given a 2D numpy array, fill colData with it."""
        assert type(data) is np.ndarray
        assert data.shape[1] == self.nCols
        for i in range(self.nCols):
            self.colData[i]=data[:,i].tolist()