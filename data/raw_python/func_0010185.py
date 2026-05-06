def isCCW(self, b, c, axis='z'):
        '''
        :b: Point or point equivalent
        :c: Point or point equivalent
        :axis: optional string or integer in set('x',0,'y',1,'z',2)
        :return: boolean

        True if the angle determined by a,self,b around 'axis'
        describes a counter-clockwise rotation, otherwise False.

        Raises CollinearPoints if self, b, c are collinear.
        '''

        result = self.ccw(b, c, axis)

        if result == 0:
            raise CollinearPoints(b, self, c)

        return result > 0