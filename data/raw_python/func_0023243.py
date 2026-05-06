def _merge_intervals(self, min_depth):
        """
        Merge overlapping intervals.

        This method is called only once in the constructor.
        """
        def add_interval(ret, start, stop):
            if min_depth is not None:
                shift = 2 * (29 - min_depth)
                mask = (int(1) << shift) - 1

                if stop - start < mask:
                    ret.append((start, stop))
                else:
                    ofs = start & mask
                    st = start
                    if ofs > 0:
                        st = (start - ofs) + (mask + 1)
                        ret.append((start, st))

                    while st + mask + 1 < stop:
                        ret.append((st, st + mask + 1))
                        st = st + mask + 1

                    ret.append((st, stop))
            else:
                ret.append((start, stop))

        ret = []
        start = stop = None
        # Use numpy sort method
        self._intervals.sort(axis=0)
        for itv in self._intervals:
            if start is None:
                start, stop = itv
                continue

            # gap between intervals
            if itv[0] > stop:
                add_interval(ret, start, stop)
                start, stop = itv
            else:
                # merge intervals
                if itv[1] > stop:
                    stop = itv[1]

        if start is not None and stop is not None:
            add_interval(ret, start, stop)

        self._intervals = np.asarray(ret)