def randomLocation(cls, radius, width, height, origin=None):
        '''
        :param: radius - float
        :param: width  - float
        :param: height - float
        :param: origin - optional Point subclass
        :return: Rectangle
        '''
        return cls(width,
                   height,
                   Point.randomLocation(radius, origin))