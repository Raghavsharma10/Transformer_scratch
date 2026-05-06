def add_classes(self, classes):
        """
        Helper to rename the classes, if provided by a dict keyed in by the orignal keys

        Parameters
        ----------
        classes : dict
            Dict of class named keyed in by sample IDs.

        Raises
        ------
        TypeError
            If classes is not a dict.
        ValueError
            If all samples in dataset are not present in input dict,
            or one of they samples in input is not recognized.

        """
        if not isinstance(classes, dict):
            raise TypeError('Input classes is not a dict!')
        if not len(classes) == self.num_samples:
            raise ValueError('Too few items - need {} keys'.format(self.num_samples))
        if not all([key in self.keys for key in classes]):
            raise ValueError('One or more unrecognized keys!')
        self.__classes = classes