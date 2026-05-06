def randomSizeAndLocation(cls, radius, widthLimits,
                              heightLimits, origin=None):
        '''
        :param: radius       - float
        :param: widthLimits  - iterable of floats with length >= 2
        :param: heightLimits - iterable of floats with length >= 2
        :param: origin       - optional Point subclass
        :return: Rectangle
        '''

        r = cls(widthLimits, heightLimits, origin)

        r.origin = Point.randomLocation(radius, origin)