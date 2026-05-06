def add(self, path):
        """
        Add the path of a data set to the list of available sets

        NOTE: a data set is assumed to be a pickled
        and gzip compressed Pandas DataFrame

        Parameters
        ----------
        path : str
        """
        name_with_ext = os.path.split(path)[1]  # split directory and filename
        name = name_with_ext.split('.')[0]  # remove extension
        self.list.update({name: path})