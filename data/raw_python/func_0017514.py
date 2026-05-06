def evaluate_at_x(self, arg, def_val=0):
        """
        Returns Y value at arg of self. Arg can be a scalar,
        but also might be np.array or other iterable
        (like list). If domain of self is not wide enough to
        interpolate the value of Y, method will return
        def_val for those arguments instead.

        Check the interpolation when arg in domain of self:
        >>> Curve([[0, 0], [2, 2], [4, 4]]).evaluate_at_x([1, 2 ,3])
        array([1., 2., 3.])

        Check if behavior of the method is correct when arg
        id partly outside the domain:
        >>> Curve([[0, 0], [2, 2], [4, 4]]).evaluate_at_x(\
            [-1, 1, 2 ,3, 5], 100)
        array([100.,   1.,   2.,   3., 100.])

        :param arg: x-value to calculate Y (may be an array or list as well)
        :param def_val: default value to return if can't interpolate at arg
        :return: np.array of Y-values at arg. If arg is a scalar,
            will return scalar as well
        """
        y = np.interp(arg, self.x, self.y, left=def_val, right=def_val)
        return y