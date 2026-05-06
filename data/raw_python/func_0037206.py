def calculate_intervals(chunk_sizes):
        """Calculate intervals for a given chunk sizes.

        :param list chunk_sizes: List of chunk sizes.
        :return: Tuple of intervals.
        :rtype: :py:class:`tuple`
        """
        start_indexes = [sum(chunk_sizes[:i]) for i in range(0, len(chunk_sizes))]
        end_indexes = [sum(chunk_sizes[:i+1]) for i in range(0, len(chunk_sizes))]
        return tuple(zip(start_indexes, end_indexes))