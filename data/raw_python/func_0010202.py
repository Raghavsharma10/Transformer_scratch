def b(self):
        '''
        Positive antipodal point on the minor axis, Point class.

        '''
        b = Point(self.center)

        if self.xAxisIsMinor:
            b.x += self.minorRadius
        else:
            b.y += self.minorRadius
        return b