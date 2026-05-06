def check_consistent_parameter_grouping(self):
        """
        Ensures this object does not have conflicting groups of parameters.

        :raises ValueError: For conflicting or absent parameters.
        """
        parameter_groups = {}
        if self.indices_per_axis is not None:
            parameter_groups["indices_per_axis"] = \
                {"self.indices_per_axis": self.indices_per_axis}
        if (self.split_size is not None) or (self.split_num_slices_per_axis is not None):
            parameter_groups["split_size"] = \
                {
                    "self.split_size": self.split_size,
                    "self.split_num_slices_per_axis": self.split_num_slices_per_axis,
            }
        if self.tile_shape is not None:
            parameter_groups["tile_shape"] = \
                {"self.tile_shape": self.tile_shape}
        if self.max_tile_bytes is not None:
            parameter_groups["max_tile_bytes"] = \
                {"self.max_tile_bytes": self.max_tile_bytes}
        if self.max_tile_shape is not None:
            if "max_tile_bytes" not in parameter_groups.keys():
                parameter_groups["max_tile_bytes"] = {}
            parameter_groups["max_tile_bytes"]["self.max_tile_shape"] = self.max_tile_shape
        if self.sub_tile_shape is not None:
            if "max_tile_bytes" not in parameter_groups.keys():
                parameter_groups["max_tile_bytes"] = {}
            parameter_groups["max_tile_bytes"]["self.sub_tile_shape"] = self.sub_tile_shape

        self.logger.debug("parameter_groups=%s", parameter_groups)

        if len(parameter_groups.keys()) > 1:
            group_keys = sorted(parameter_groups.keys())
            raise ValueError(
                "Got conflicting parameter groups specified, "
                +
                "should only specify one group to define the split:\n"
                +
                (
                    "\n".join(
                        [
                            (
                                ("Group %18s: " % ("'%s'" % group_key))
                                +
                                str(parameter_groups[group_key])
                            )
                            for group_key in group_keys
                        ]
                    )
                )
            )
        if len(parameter_groups.keys()) <= 0:
            raise ValueError(
                "No split parameters specified, need parameters from one of the groups: "
                +
                "'indices_per_axis', 'split_size', 'tile_shape' or 'max_tile_bytes'"
            )