def check_consistent_parameter_dimensions(self):
        """
        Ensure that all parameter dimensions are consistent with
        the :attr:`array_shape` dimension.

        :raises ValueError: For inconsistent parameter dimensions.
        """
        if self.indices_per_axis is not None:
            if len(self.indices_per_axis) > len(self.array_shape):
                raise ValueError(
                    "Got len(self.indices_per_axis)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.indices_per_axis), len(self.array_shape))
                )
        if self.split_num_slices_per_axis is not None:
            if len(self.split_num_slices_per_axis) > len(self.array_shape):
                raise ValueError(
                    (
                        "Got len(self.split_num_slices_per_axis)=%s > len(self.array_shape)=%s,"
                        +
                        " should be equal."
                    )
                    %
                    (len(self.split_num_slices_per_axis), len(self.array_shape))
                )
        if self.tile_shape is not None:
            if len(self.tile_shape) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.tile_shape)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.tile_shape), len(self.array_shape))
                )

        if self.sub_tile_shape is not None:
            if len(self.sub_tile_shape) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.sub_tile_shape)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.sub_tile_shape), len(self.array_shape))
                )

        if self.max_tile_shape is not None:
            if len(self.max_tile_shape) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.max_tile_shape)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.max_tile_shape), len(self.array_shape))
                )

        if self.array_start is not None:
            if len(self.array_start) != len(self.array_shape):
                raise ValueError(
                    "Got len(self.array_start)=%s > len(self.array_shape)=%s, should be equal."
                    %
                    (len(self.array_start), len(self.array_shape))
                )