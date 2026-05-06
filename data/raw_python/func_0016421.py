def ratio_deep_compare(self, other, settings):
        """
        Compares each field of the name one at a time to see if they match.
        Each name field has context-specific comparison logic.

        :param Name other: other Name for comparison
        :return int: sequence ratio match (out of 100)
        """

        if not self._is_compatible_with(other):
            return 0

        first, middle, last = self._compare_components(other, settings, True)
        f_weight, m_weight, l_weight = self._determine_weights(other, settings)
        total_weight = f_weight + m_weight + l_weight

        result = (
            first * f_weight +
            middle * m_weight +
            last * l_weight
        ) / total_weight

        return result