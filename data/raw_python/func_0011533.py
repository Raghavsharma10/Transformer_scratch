def _update_consumed_ranges(self, start_pos, end_pos):
        """Update the ``self.consumed_ranges`` array with which
        byte ranges have been consecutively consumed.
        """
        self.range_set.add(Interval(start_pos, end_pos+1))
        self.range_set.merge_overlaps()