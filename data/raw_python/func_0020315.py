def body(self):
        '''
        Returns the axis instance where the light curves will be shown

        '''

        res = self._body[self.bcount]()
        self.bcount += 1
        return res