def xAxisIsMinor(self):
        '''
        Returns True if the minor axis is parallel to the X axis, boolean.
        '''
        return min(self.radius.x, self.radius.y) == self.radius.x