def right(self):
        '''
        Returns the current axis instance on the right side of the
        page, where cross-validation information is displayed

        '''

        res = self.body_right[self.rcount]()
        self.rcount += 1
        return res