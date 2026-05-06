def b_neg(self):
        '''
        Negative antipodal point on the minor axis, Point class.
        '''
        nb = Point(self.center)

        if self.xAxisIsMinor:
            nb.x -= self.minorRadius
        else:
            nb.y -= self.minorRadius
        return nb