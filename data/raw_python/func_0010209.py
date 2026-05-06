def normal(self):
        '''
        :return: Line

        Returns a Line normal (perpendicular) to this Line.
        '''

        d = self.B - self.A

        return Line([-d.y, d.x], [d.y, -d.x])