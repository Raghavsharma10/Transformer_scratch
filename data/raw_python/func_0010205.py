def focus0(self):
        '''
        First focus of the ellipse, Point class.

        '''
        f = Point(self.center)

        if self.xAxisIsMajor:
            f.x -= self.linearEccentricity
        else:
            f.y -= self.linearEccentricity
        return f