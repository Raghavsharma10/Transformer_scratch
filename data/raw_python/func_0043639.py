def set_split_extents_by_indices_per_axis(self):
        """
        Sets split shape :attr:`split_shape` and
        split extents (:attr:`split_begs` and :attr:`split_ends`)
        from values in :attr:`indices_per_axis`.
        """
        if self.indices_per_axis is None:
            raise ValueError("Got None for self.indices_per_axis")

        self.logger.debug("self.array_shape=%s", self.array_shape)
        self.logger.debug("self.indices_per_axis=%s", self.indices_per_axis)
        self.indices_per_axis = \
            pad_with_none(self.indices_per_axis, len(self.array_shape))

        # Define the start and stop indices (extents) for each axis slice
        self.split_shape = _np.ones(len(self.array_shape), dtype="int64")
        self.split_begs = [[], ] * len(self.array_shape)
        self.split_ends = [[], ] * len(self.array_shape)
        for i in range(len(self.indices_per_axis)):
            indices = self.indices_per_axis[i]
            if (indices is not None) and (len(indices) > 0):
                self.split_shape[i] = len(indices) + 1
                self.split_begs[i] = _np.zeros((len(indices) + 1,), dtype="int64")
                self.split_begs[i][1:] = indices
                self.split_ends[i] = _np.zeros((len(self.split_begs[i]),), dtype="int64")
                self.split_ends[i][0:-1] = self.split_begs[i][1:]
                self.split_ends[i][-1] = self.array_shape[i]
            else:
                # start and stop is the full width of the axis
                self.split_begs[i] = [0, ]
                self.split_ends[i] = [self.array_shape[i], ]

        self.logger.debug("self.indices_per_axis=%s", self.indices_per_axis)