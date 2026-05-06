def split_by_percent(self, spin_systems_list):
        """Split list of spin systems by specified percentages.

        :param list spin_systems_list: List of spin systems.
        :return: List of spin systems divided into sub-lists corresponding to specified split percentages.
        :rtype: :py:class:`list`
        """
        chunk_sizes = [int((i*len(spin_systems_list))/100) for i in self.plsplit]
        if sum(chunk_sizes) < len(spin_systems_list):
            difference = len(spin_systems_list) - sum(chunk_sizes)
            chunk_sizes[chunk_sizes.index(min(chunk_sizes))] += difference

        assert sum(chunk_sizes) == len(spin_systems_list), \
            "sum of chunk sizes must be equal to spin systems list length."

        intervals = self.calculate_intervals(chunk_sizes)
        chunks_of_spin_systems_by_percentage = [itertools.islice(spin_systems_list, *interval) for interval in intervals]
        return chunks_of_spin_systems_by_percentage