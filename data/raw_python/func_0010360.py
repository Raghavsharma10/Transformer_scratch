def addPlateau(self, plateauAmount, pitchSampFreq=None):
        '''
        Add a plateau
        
        A negative plateauAmount will move the peak backwards.
        A positive plateauAmount will move the peak forwards.
        
        All points on the side of the peak growth will also get moved.
        i.e. the slope of the peak does not change.  The accent gets
        wider instead.
        
        If pitchSampFreq=None, the plateau will only be specified by
        the start and end points of the plateau
        '''
        if plateauAmount == 0:
            return
        
        maxPoint = self.pointList[self.peakI]
        
        # Define the plateau
        if pitchSampFreq is not None:
            numSteps = abs(int(plateauAmount / pitchSampFreq))
            timeChangeList = [stepV * pitchSampFreq
                              for stepV in
                              range(0, numSteps + 1)]
        else:
            timeChangeList = [plateauAmount, ]
            
        # Shift the side being pushed by the plateau
        if plateauAmount < 0:  # Plateau moves left of the peak
            leftSide = self.pointList[:self.peakI]
            rightSide = self.pointList[self.peakI:]
            
            plateauPoints = [(maxPoint[0] + timeChange, maxPoint[1])
                             for timeChange in timeChangeList]
            leftSide = [(timeV + plateauAmount, f0V)
                        for timeV, f0V in leftSide]
            self.netLeftShift += plateauAmount
            
        elif plateauAmount > 0:  # Plateau moves right of the peak
            leftSide = self.pointList[:self.peakI + 1]
            rightSide = self.pointList[self.peakI + 1:]
            
            plateauPoints = [(maxPoint[0] + timeChange, maxPoint[1])
                             for timeChange in timeChangeList]
            rightSide = [(timeV + plateauAmount, f0V)
                         for timeV, f0V in rightSide]
            self.netRightShift += plateauAmount
        
        self.pointList = leftSide + plateauPoints + rightSide