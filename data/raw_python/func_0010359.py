def adjustPeakHeight(self, heightAmount):
        '''
        Adjust peak height
        
        The foot of the accent is left unchanged and intermediate
        values are linearly scaled
        '''
        if heightAmount == 0:
            return
        
        pitchList = [f0V for _, f0V in self.pointList]
        minV = min(pitchList)
        maxV = max(pitchList)
        scale = lambda x, y: x + y * (x - minV) / float(maxV - minV)
        
        self.pointList = [(timeV, scale(f0V, heightAmount))
                          for timeV, f0V in self.pointList]