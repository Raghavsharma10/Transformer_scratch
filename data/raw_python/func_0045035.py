def _data(self, copy=False):
        """
        Get data kwargs of the container (i.e. dataframe and series objects).
        """
        data = {}
        for key, obj in vars(self).items():
            if isinstance(obj, (pd.Series, pd.DataFrame, pd.SparseSeries, pd.SparseDataFrame)):
                if copy:
                    data[key] = obj.copy(deep=True)
                else:
                    data[key] = obj
        return data