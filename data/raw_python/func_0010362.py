def deleteOverlapping(self, targetList):
        '''
        Erase points from another list that overlap with points in this list
        '''
        start = self.pointList[0][0]
        stop = self.pointList[-1][0]
        
        if self.netLeftShift < 0:
            start += self.netLeftShift
            
        if self.netRightShift > 0:
            stop += self.netRightShift
            
        targetList = _deletePoints(targetList, start, stop)
        
        return targetList