def left(self):
        '''
        Returns the current axis instance on the left side of
        the page where each successive light curve is displayed

        '''

        res = self.body_left[self.lcount]()
        self.lcount += 1
        return res