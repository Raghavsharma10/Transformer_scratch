def _add_interval(all_intervals, new_interval):
        """
        Adds a new interval to a set of none overlapping intervals.

        :param set[(int,int)] all_intervals: The set of distinct intervals.
        :param (int,int) new_interval: The new interval.
        """
        intervals = None
        old_interval = None
        for old_interval in all_intervals:
            intervals = Type2CondenseHelper._distinct(new_interval, old_interval)
            if intervals:
                break

        if intervals is None:
            all_intervals.add(new_interval)
        else:
            if old_interval:
                all_intervals.remove(old_interval)
            for distinct_interval in intervals:
                Type2CondenseHelper._add_interval(all_intervals, distinct_interval)