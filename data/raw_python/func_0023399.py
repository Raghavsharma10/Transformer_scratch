def consistency(self):
        """
        Get a percentage of fill between the min and max time the moc is defined.

        A value near 0 shows a sparse temporal moc (i.e. the moc does not cover a lot
        of time and covers very distant times. A value near 1 means that the moc covers
        a lot of time without big pauses.

        Returns
        -------
        result : float
            fill percentage (between 0 and 1.)

        """

        result = self.total_duration.jd / (self.max_time - self.min_time).jd
        return result