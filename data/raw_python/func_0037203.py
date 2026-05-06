def seq_site_length(self):
        """Calculate length of a single sequence site based upon relative positions specified in peak descriptions.

        :return: Length of sequence site.
        :rtype: :py:class:`int`
        """
        relative_positions_set = set()
        for peak_descr in self:
            relative_positions_set.update(peak_descr.relative_positions)
        return len(relative_positions_set)