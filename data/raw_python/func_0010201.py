def a_neg(self):
        '''
        Negative antipodal point on the major axis, Point class.

        '''
        na = Point(self.center)

        if self.xAxisIsMajor:
            na.x -= self.majorRadius
        else:
            na.y -= self.majorRadius
        return na