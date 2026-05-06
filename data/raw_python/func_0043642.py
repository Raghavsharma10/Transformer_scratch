def set_split_extents_by_split_size(self):
        """
        Sets split shape :attr:`split_shape` and
        split extents (:attr:`split_begs` and :attr:`split_ends`)
        from values in :attr:`split_size` and :attr:`split_num_slices_per_axis`.
        """

        if self.split_size is None:
            if (
                _np.all([s is not None for s in self.split_num_slices_per_axis])
                and
                _np.all([s > 0 for s in self.split_num_slices_per_axis])
            ):
                self.split_size = _np.product(self.split_num_slices_per_axis)
            else:
                raise ValueError(
                    (
                        "Got invalid self.split_num_slices_per_axis=%s, all elements "
                        +
                        "need to be integers greater than zero when self.split_size is None."
                    )
                    %
                    self.split_num_slices_per_axis
                )
        self.logger.debug(
            "Pre  cannonicalise: self.split_num_slices_per_axis=%s",
            self.split_num_slices_per_axis)
        self.split_num_slices_per_axis = \
            calculate_num_slices_per_axis(
                self.split_num_slices_per_axis,
                self.split_size,
                self.array_shape
            )
        self.logger.debug(
            "Post cannonicalise: self.split_num_slices_per_axis=%s",
            self.split_num_slices_per_axis)
        # Define the start and stop indices (extents) for each axis slice
        self.split_shape = self.split_num_slices_per_axis.copy()
        self.split_begs = [[], ] * len(self.array_shape)
        self.split_ends = [[], ] * len(self.array_shape)
        for i in range(len(self.array_shape)):
            self.split_begs[i], self.split_ends[i] = \
                self.calculate_axis_split_extents(
                    self.split_shape[i],
                    self.array_shape[i]
            )