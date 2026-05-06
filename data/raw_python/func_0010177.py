def units(cls, scale=1):
        '''
        :scale: optional integer scaling factor
        :return: list of three Point subclass

        Returns three points whose coordinates are the head of a
        unit vector from the origin ( conventionally i, j and k).

        '''
        return [cls(x=scale), cls(y=scale), cls(z=scale)]