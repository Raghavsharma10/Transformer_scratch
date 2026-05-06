def randomSize(cls, widthLimits, heightLimits, origin=None):
        '''
        :param: widthLimits  - iterable of integers with length >= 2
        :param: heightLimits - iterable of integers with length >= 2
        :param: origin       - optional Point subclass
        :return: Rectangle
        '''

        r = cls(0, 0, origin)

        r.w = random.randint(widthLimits[0], widthLimits[1])
        r.h = random.randint(heightLimits[0], heightLimits[1])

        return r