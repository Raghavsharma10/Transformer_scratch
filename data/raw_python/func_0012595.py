def filter(self, index): #@ReservedAssignment
        """
        Filters a datamat by different aspects.

        This function is a device to filter the datamat by certain logical
        conditions. It takes as input a logical array (contains only True
        or False for every datapoint) and kicks out all datapoints for which
        the array says False. The logical array can conveniently be created
        with numpy::

            >>> print np.unique(fm.category)
            np.array([2,9])
            >>> fm_filtered = fm[ fm.category == 9 ]
            >>> print np.unique(fm_filtered)
            np.array([9])

        Parameters:
            index : array
                Array-like that contains True for every element that
                passes the filter; else contains False
        Returns:
            datamat : Datamat Instance
        """
        return Datamat(categories=self._categories, datamat=self, index=index)