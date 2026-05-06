def contains(self, times, keep_inside=True, delta_t=DEFAULT_OBSERVATION_TIME):
        """
        Get a mask array (e.g. a numpy boolean array) of times being inside (or outside) the
        TMOC instance.

        Parameters
        ----------
        times : `astropy.time.Time`
            astropy times to check whether they are contained in the TMOC or not.
        keep_inside : bool, optional
            True by default. If so the filtered table contains only observations that are located the MOC.
            If ``keep_inside`` is False, the filtered table contains all observations lying outside the MOC.
        delta_t : `astropy.time.TimeDelta`, optional
            the duration of one observation. It is set to 30 min by default. This data is used to compute the
            more efficient TimeMOC order to represent the observations (Best order = the less precise order which
            is able to discriminate two observations separated by ``delta_t``).

        Returns
        -------
        array : `~numpy.darray`
            A mask boolean array
        """
        # the requested order for filtering the astropy observations table is more precise than the order
        # of the TimeMoc object
        current_max_order = self.max_order
        new_max_order = TimeMOC.time_resolution_to_order(delta_t)
        if new_max_order > current_max_order:
            message = 'Requested time resolution filtering cannot be applied.\n' \
                      'Filtering is applied with a time resolution of {0} sec.'.format(
                TimeMOC.order_to_time_resolution(current_max_order).sec)
            warnings.warn(message, UserWarning)

        rough_tmoc = self.degrade_to_order(new_max_order)

        pix_arr = (times.jd * TimeMOC.DAY_MICRO_SEC)
        pix_arr = pix_arr.astype(int)

        intervals_arr = rough_tmoc._interval_set._intervals
        inf_arr = np.vstack([pix_arr[i] >= intervals_arr[:, 0] for i in range(pix_arr.shape[0])])
        sup_arr = np.vstack([pix_arr[i] <= intervals_arr[:, 1] for i in range(pix_arr.shape[0])])

        if keep_inside:
            res = inf_arr & sup_arr
            filtered_rows = np.any(res, axis=1)
        else:
            res = ~inf_arr | ~sup_arr
            filtered_rows = np.all(res, axis=1)

        return filtered_rows