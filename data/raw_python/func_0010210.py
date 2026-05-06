def t(self, point):
        '''
        :point: Point subclass
        :return: float

        If :point: is collinear, determine the 't' coefficient of
        the parametric equation:

        xyz = A<xyz> + t ( B<xyz> - A<xyz> )

        if t < 0, point is less than A and B on the line
        if t >= 0 and <= 1, point is between A and B
        if t > 1 point is greater than B
        '''

        # XXX could use for an ordering on points?

        if point not in self:
            msg = "'{p}' is not collinear with '{l}'"
            raise CollinearPoints(msg.format(p=point, l=self))

        # p = A + t ( B - A)
        # p - A = t ( B - A)
        # p - A / (B -A) = t

        return (point - self.A) / self.m