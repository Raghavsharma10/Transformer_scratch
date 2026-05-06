def yAxisIsMajor(self):
        '''
        Returns True if the major axis is parallel to the Y axis, boolean.
        '''
        return max(self.radius.x, self.radius.y) == self.radius.y