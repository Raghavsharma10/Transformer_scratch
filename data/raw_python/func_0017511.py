def rescale(self, factor=1.0, allow_cast=True):
        """
        Rescales self.y by given factor, if allow_cast is set to True
        and division in place is impossible - casting and not in place
        division may occur occur. If in place is impossible and allow_cast
        is set to False - an exception is raised.

        Check simple rescaling by 2 with no casting
        >>> c = Curve([[0, 0], [5, 5], [10, 10]], dtype=np.float)
        >>> c.rescale(2, allow_cast=False)
        >>> print(c.y)
        [0.  2.5 5. ]

        Check rescaling with floor division
        >>> c = Curve([[0, 0], [5, 5], [10, 10]], dtype=np.int)
        >>> c.rescale(1.5, allow_cast=True)
        >>> print(c.y)
        [0 3 6]

        >>> c = Curve([[0, 0], [5, 5], [10, 10]], dtype=np.int)
        >>> c.rescale(-1, allow_cast=True)
        >>> print(c.y)
        [  0  -5 -10]

        :param factor: rescaling factor, should be a number
        :param allow_cast: bool - allow division not in place
        """
        try:
            self.y /= factor
        except TypeError as e:
            logger.warning("Division in place is impossible: %s", e)
            if allow_cast:
                self.y = self.y / factor
            else:
                logger.error("allow_cast flag set to True should help")
                raise