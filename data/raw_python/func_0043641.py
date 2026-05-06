def calculate_axis_split_extents(self, num_sections, size):
        """
        Divides :samp:`range(0, {size})` into (approximately) equal sized
        intervals. Returns :samp:`(begs, ends)` where :samp:`slice(begs[i], ends[i])`
        define the intervals for :samp:`i in range(0, {num_sections})`.

        :type num_sections: :obj:`int`
        :param num_sections: Divide  :samp:`range(0, {size})` into this
           many intervals (approximately) equal sized intervals.
        :type size: :obj:`int`
        :param size: Range for the subdivision.
        :rtype: :obj:`tuple`
        :return: Two element tuple :samp:`(begs, ends)`
           such that :samp:`slice(begs[i], ends[i])` define the
           intervals for :samp:`i in range(0, {num_sections})`.

        """
        section_size = size // num_sections
        if section_size >= 1:
            begs = _np.arange(0, section_size * num_sections, section_size)
            rem = size - section_size * num_sections
            if rem > 0:
                for i in range(rem):
                    begs[i + 1:] += 1
            ends = _np.zeros_like(begs)
            ends[0:-1] = begs[1:]
            ends[-1] = size
        else:
            begs = _np.arange(0, num_sections)
            begs[size:] = size
            ends = begs.copy()
            ends[0:-1] = begs[1:]

        return begs, ends