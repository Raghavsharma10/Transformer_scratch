def split(self, indices):
        """
        Splits logical networks according to given indices

        Parameters
        ----------
        indices : list
            1-D array of sorted integers, the entries indicate where the array is split

        Returns
        -------
        list
            List of :class:`caspo.core.logicalnetwork.LogicalNetworkList` object instances


        .. seealso:: `numpy.split <http://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html#numpy-split>`_
        """
        return [LogicalNetworkList(self.hg, part) for part in np.split(self.__matrix, indices)]