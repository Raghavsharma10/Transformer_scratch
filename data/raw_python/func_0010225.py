def scale(self, dx=1.0, dy=1.0):
        '''
        :param: dx - optional float
        :param: dy - optional float

        Scales the rectangle's width and height by dx and dy.

        '''
        self.width *= dx
        self.height *= dy