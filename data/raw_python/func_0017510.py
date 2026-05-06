def normalize(self, dt, allow_cast=True):
        """
        Normalize to 1 over [-dt, +dt] area, if allow_cast is set
        to True, division not in place and casting may occur.
        If division in place is not possible and allow_cast is False
        an exception is raised.

        >>> a = Profile([[0, 0], [1, 5], [2, 10], [3, 5], [4, 0]])
        >>> a.normalize(1, allow_cast=True)
        >>> print(a.y)
        [0. 2. 4. 2. 0.]

        :param dt:
        :param allow_cast:
        """
        if dt <= 0:
            raise ValueError("Expected positive input")
        logger.info('Running %(name)s.normalize(dt=%(dt)s)', {"name": self.__class__, "dt": dt})
        try:
            ave = np.average(self.y[np.fabs(self.x) <= dt])
        except RuntimeWarning as e:
            logger.error('in normalize(). self class is %(name)s, dt=%(dt)s', {"name": self.__class__, "dt": dt})
            raise Exception("Scaling factor error: {0}".format(e))
        try:
            self.y /= ave
        except TypeError as e:
            logger.warning("Division in place is impossible: %s", e)
            if allow_cast:
                self.y = self.y / ave
            else:
                logger.error("Division in place impossible - allow_cast flag set to True should help")
                raise