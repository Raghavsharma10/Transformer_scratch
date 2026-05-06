def yAxisIsMinor(self):
        '''
        Returns True if the minor axis is parallel to the Y axis, boolean.
        '''
        return min(self.radius.x, self.radius.y) == self.radius.y