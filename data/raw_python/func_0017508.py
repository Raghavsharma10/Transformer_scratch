def x_at_y(self, y, reverse=False):
        """
        Calculates inverse profile - for given y returns x such that f(x) = y
        If given y is not found in the self.y, then interpolation is used.
        By default returns first result looking from left,
        if reverse argument set to True,
        looks from right. If y is outside range of self.y
        then np.nan is returned.

        Use inverse lookup to get x-coordinate of first point:
        >>> float(Profile([[0.0, 5.0], [0.1, 10.0], [0.2, 20.0], [0.3, 10.0]])\
            .x_at_y(5.))
        0.0

        Use inverse lookup to get x-coordinate of second point,
        looking from left:
        >>> float(Profile([[0.0, 5.0], [0.1, 10.0], [0.2, 20.0], [0.3, 10.0]])\
            .x_at_y(10.))
        0.1

        Use inverse lookup to get x-coordinate of fourth point,
        looking from right:
        >>> float(Profile([[0.0, 5.0], [0.1, 10.0], [0.2, 20.0], [0.3, 10.0]])\
            .x_at_y(10., reverse=True))
        0.3

        Use interpolation between first two points:
        >>> float(Profile([[0.0, 5.0], [0.1, 10.0], [0.2, 20.0], [0.3, 10.0]])\
            .x_at_y(7.5))
        0.05

        Looking for y below self.y range:
        >>> float(Profile([[0.0, 5.0], [0.1, 10.0], [0.2, 20.0], [0.3, 10.0]])\
            .x_at_y(2.0))
        nan

        Looking for y above self.y range:
        >>> float(Profile([[0.0, 5.0], [0.1, 10.0], [0.2, 20.0], [0.3, 10.0]])\
            .x_at_y(22.0))
        nan

        :param y: reference value
        :param reverse: boolean value - direction of lookup
        :return: x value corresponding to given y or NaN if not found
        """
        logger.info('Running %(name)s.y_at_x(y=%(y)s, reverse=%(rev)s)',
                    {"name": self.__class__, "y": y, "rev": reverse})
        # positive or negative direction handles
        x_handle, y_handle = self.x, self.y
        if reverse:
            x_handle, y_handle = self.x[::-1], self.y[::-1]

        # find the index of first value in self.y greater or equal than y
        cond = y_handle >= y
        ind = np.argmax(cond)

        # two boundary conditions where x cannot be found:
        # A) y > max(self.y)
        # B) y < min(self.y)

        # A) if y > max(self.y) then condition self.y >= y
        #   will never be satisfied
        #   np.argmax( cond ) will be equal 0  and  cond[ind] will be False
        if not cond[ind]:
            return np.nan

        # B) if y < min(self.y) then condition self.y >= y
        #   will be satisfied on first item
        #   np.argmax(cond) will be equal 0,
        #   to exclude situation that y_handle[0] = y
        #   we also check if y < y_handle[0]
        if ind == 0 and y < y_handle[0]:
            return np.nan

        # use lookup if y in self.y:
        if cond[ind] and y_handle[ind] == y:
            return x_handle[ind]

        # alternatively - pure python implementation
        # return x_handle[ind] - \
        #        ((x_handle[ind] - x_handle[ind - 1]) / \
        #        (y_handle[ind] - y_handle[ind - 1])) * \
        #        (y_handle[ind] - y)

        # use interpolation
        sl = slice(ind - 1, ind + 1)
        return np.interp(y, y_handle[sl], x_handle[sl])