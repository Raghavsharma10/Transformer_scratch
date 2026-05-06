def xAxisIsMajor(self):
        '''
        Returns True if the major axis is parallel to the X axis, boolean.
        '''
        return max(self.radius.x, self.radius.y) == self.radius.x