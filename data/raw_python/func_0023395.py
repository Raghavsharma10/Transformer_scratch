def remove_neighbours(self):
        """
        Remove all the pixels at max order located at the bound of the moc

        """
        time_delta = 1 << (2*(IntervalSet.HPY_MAX_ORDER - self.max_order))

        intervals_arr = self._interval_set._intervals
        intervals_arr[:, 0] = np.minimum(intervals_arr[:, 0] + time_delta, (1 << 58) - 1)
        intervals_arr[:, 1] = np.maximum(intervals_arr[:, 1] - time_delta, 0)

        good_intervals = intervals_arr[:, 1] > intervals_arr[:, 0]

        self._interval_set = IntervalSet(intervals_arr[good_intervals])