def withAngles(cls, origin=None, base=1, alpha=None,
                   beta=None, gamma=None, inDegrees=False):
        '''
        :origin: optional Point
        :alpha: optional float describing length of the side opposite A
        :beta: optional float describing length of the side opposite B
        :gamma: optional float describing length of the side opposite C
        :return: Triangle initialized with points comprising the triangle
                 with the specified angles.
        '''
        raise NotImplementedError("withAngles")