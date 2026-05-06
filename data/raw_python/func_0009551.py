def MTF50(self, MTFx,MTFy):
        '''
        return object resolution as [line pairs/mm]
               where MTF=50%
               see http://www.imatest.com/docs/sharpness/
        '''
        if self.mtf_x is None:
            self.MTF()
        f = UnivariateSpline(self.mtf_x, self.mtf_y-0.5)
        return f.roots()[0]