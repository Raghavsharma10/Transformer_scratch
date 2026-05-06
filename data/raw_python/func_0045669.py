def _data(self, copy=False):
        """
        Get all data associated with the container as key value pairs.
        """
        data = {}
        for key, obj in self.__dict__.items():
            if isinstance(obj, (pd.Series, pd.DataFrame, pd.SparseSeries, pd.SparseDataFrame)):
                if copy:
                    data[key] = obj.copy()
                else:
                    data[key] = obj
        return data