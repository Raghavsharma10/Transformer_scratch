def reintegrate(self, fullPointList):
        '''
        Integrates the pitch values of the accent into a larger pitch contour
        '''
        # Erase the original region of the accent
        fullPointList = _deletePoints(fullPointList, self.minT, self.maxT)
        
        # Erase the new region of the accent
        fullPointList = self.deleteOverlapping(fullPointList)
        
        # Add the accent into the full pitch list
        outputPointList = fullPointList + self.pointList
        outputPointList.sort()
        
        return outputPointList