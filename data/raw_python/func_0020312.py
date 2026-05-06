def top_right(self):
        '''
        Returns the axis instance at the top right of the page,
        where the postage stamp and aperture is displayed

        '''

        res = self.body_top_right[self.tcount]()
        self.tcount += 1
        return res