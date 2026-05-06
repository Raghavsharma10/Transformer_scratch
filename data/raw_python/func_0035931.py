def get_pandas_series(self):
        """The function creates pandas series based on index and values"""
        return pandas.Series(self.values, self.index, name=self.name)