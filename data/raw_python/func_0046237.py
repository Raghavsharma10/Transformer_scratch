def resample(self, new_timepoints, extrapolate=False):

        """
        Use linear interpolation to resample trajectory values.
        The new values are interpolated for the provided time points.
        This is generally before comparing or averaging trajectories.

        :param new_timepoints: the new time points
        :param extrapolate: whether extrapolation should be performed when some new time points
            are out of the current time range. if extrapolate=False, it would raise an exception.
        :return: a new trajectory.
        :rtype: :class:`~means.simulation.trajectory.Trajectory`
        """
        if not extrapolate:
            if min(self.timepoints) > min(new_timepoints):
                raise Exception("Some of the new time points are before any time points. If you really want to extrapolate, use `extrapolate=True`")
            if max(self.timepoints) < max(new_timepoints):
                raise Exception("Some of the new time points are after any time points. If you really want to extrapolate, use `extrapolate=True`")
        new_values = np.interp(new_timepoints, self.timepoints, self.values)
        return Trajectory(new_timepoints, new_values, self.description)