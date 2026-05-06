def create_dimension_groups(dimension_positions):
        """Create list of dimension groups.

        :param zip dimension_positions: List of tuples describing dimension and its position within sequence site.
        :return: List of dimension groups.
        :rtype: :py:class:`list`
        """
        dimension_groups = []
        for dim_group_label, position in dimension_positions:
            dim_group = DimensionGroup(dim_group_label, position)

            for dim_label in nmrstarlib.RESONANCE_CLASSES[dim_group_label]:
                dim_group.dimensions.append(Dimension(dim_label, position))
            dimension_groups.append(dim_group)

        return dimension_groups