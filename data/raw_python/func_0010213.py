def intersection(self, other):
        '''
        :param: other - Line subclass
        :return: Point subclass

        Returns a Point object with the coordinates of the intersection
        between the current line and the other line.

        Will raise Parallel() if the two lines are parallel.
        Will raise Collinear() if the two lines are collinear.
        '''

        if self.isCollinear(other):
            msg = '{!r} and {!r} are collinear'
            raise CollinearLines(msg.format(self, other))

        d0 = self.A - self.B
        d1 = other.A - other.B

        denominator = (d0.x * d1.y) - (d0.y * d1.x)

        if denominator == 0:
            msg = '{!r} and {!r} are parallel'
            raise ParallelLines(msg.format(self, other))

        cp0 = self.A.cross(self.B)
        cp1 = other.A.cross(other.B)

        x_num = (cp0 * d1.x) - (d0.x * cp1)
        y_num = (cp0 * d1.y) - (d0.y * cp1)

        p = Point(x_num / denominator, y_num / denominator)

        if p in self and p in other:
            return p

        msg = "found point {!r} but not in {!r} and {!r}"
        raise ParallelLines(msg.format(p, self, other))