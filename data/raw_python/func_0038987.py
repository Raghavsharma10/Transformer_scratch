def extend(self, other):
        """
        Method to extend the dataset vertically (add samples from  anotehr dataset).

        Parameters
        ----------
        other : MLDataset
            second dataset to be combined with the current
            (different samples, but same dimensionality)

        Raises
        ------
        TypeError
            if input is not an MLDataset.
        """

        if not isinstance(other, MLDataset):
            raise TypeError('Incorrect type of dataset provided!')
        # assert self.__dtype==other.dtype, TypeError('Incorrect data type of features!')
        for sample in other.keys:
            self.add_sample(sample, other.data[sample], other.labels[sample],
                            other.classes[sample])