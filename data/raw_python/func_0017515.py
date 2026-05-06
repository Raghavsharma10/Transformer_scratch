def subtract(self, curve2, new_obj=False):
        """
        Method that calculates difference between 2 curves
        (or subclasses of curves). Domain of self must be in
        domain of curve2 what means min(self.x) >= min(curve2.x)
        and max(self.x) <= max(curve2.x).
        Might modify self, and can return the result or None

        Use subtract as -= operator, check whether returned value is None:
        >>> Curve([[0, 0], [1, 1], [2, 2], [3, 1]]).subtract(\
            Curve([[-1, 1], [5, 1]])) is None
        True

        Use subtract again but return a new object this time.
        >>> Curve([[0, 0], [1, 1], [2, 2], [3, 1]]).subtract(\
            Curve([[-1, 1], [5, 1]]), new_obj=True).y
        DataSet([-1.,  0.,  1.,  0.])

        Try using wrong inputs to create a new object,
        and check whether it throws an exception:
        >>> Curve([[0, 0], [1, 1], [2, 2], [3, 1]]).subtract(\
            Curve([[1, -1], [2, -1]]), new_obj=True) is None
        Traceback (most recent call last):
        ...
        Exception: curve2 does not include self domain


        :param curve2: second object to calculate difference
        :param new_obj: if True, method is creating new object
            instead of modifying self
        :return: None if new_obj is False (but will modify self)
            or type(self) object containing the result
        """
        # domain1 = [a1, b1]
        # domain2 = [a2, b2]
        a1, b1 = np.min(self.x), np.max(self.x)
        a2, b2 = np.min(curve2.x), np.max(curve2.x)

        # check whether domain condition is satisfied
        if a2 > a1 or b2 < b1:
            logger.error("Domain of self must be in domain of given curve")
            raise Exception("curve2 does not include self domain")
        # if we want to create and return a new object
        # rather then modify existing one
        if new_obj:
            return functions.subtract(self, curve2.change_domain(self.x))
        values = curve2.evaluate_at_x(self.x)
        self.y = self.y - values
        return None