def shiftAccent(self, shiftAmount):
        '''
        Move the whole accent earlier or later
        '''
        if shiftAmount == 0:
            return
        
        self.pointList = [(time + shiftAmount, pitch)
                          for time, pitch in self.pointList]
        
        # Update shift amounts
        if shiftAmount < 0:
            self.netLeftShift += shiftAmount
        elif shiftAmount >= 0:
            self.netRightShift += shiftAmount