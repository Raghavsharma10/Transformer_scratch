def flatFieldFromFit(self):
        '''
        calculate flatField from 2d-polynomal fit filling
        all high gradient areas within averaged fit-image

        returns flatField, average background level, fitted image, valid indices mask
        '''
        fitimg, mask = self._prepare()

        out = fitimg.copy()
        lastm = 0

        for _ in range(10):
            out = polyfit2dGrid(out, mask, 2)
            mask = highGrad(out)
            m = mask.sum()
            if m == lastm:
                break
            lastm = m

        out = np.clip(out, 0.1, 1)

        out = resize(out, self._orig_shape, mode='reflect')
        return out, self.bglevel / self._n, fitimg, mask