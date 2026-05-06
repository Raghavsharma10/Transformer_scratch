def stdDev(self):
        '''
        get the standard deviation 
        from the PSF is evaluated as 2d Gaussian
        '''
        if self._corrPsf is None:
            self.psf()
        p = self._corrPsf.copy()
        mn = p.min()
        p[p<0.05*p.max()] = mn
        p-=mn
        p/=p.sum()
        
        x,y = self._psfGridCoords()
        x = x.flatten()
        y = y.flatten()

        guess = (1,1,0)

        param, _ = curve_fit(self._fn, (x,y), p.flatten(), guess)

        self._fitParam = param 
        stdx,stdy =  param[:2]
        self._std = (stdx+stdy) / 2
        
        return self._std