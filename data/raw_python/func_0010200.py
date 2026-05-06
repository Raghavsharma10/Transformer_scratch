def a(self):
        '''
        Positive antipodal point on the major axis, Point class.

        '''
        a = Point(self.center)

        if self.xAxisIsMajor:
            a.x += self.majorRadius
        else:
            a.y += self.majorRadius
        return a