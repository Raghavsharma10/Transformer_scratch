def reset_index(self):
        """
        Resets the index of the Series to simple integer list and the index name to 'index'.

        :return: nothing
        """
        self.index = list(range(self.__len__()))
        self.index_name = 'index'