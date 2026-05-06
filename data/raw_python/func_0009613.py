def flatFieldFromFunction(self):
        '''
        calculate flatField from fitting vignetting function to averaged fit-image
        returns flatField, average background level, fitted image, valid indices mask
        '''
        fitimg, mask = self._prepare()
        mask = ~mask

        s0, s1 = fitimg.shape
        #f-value, alpha, fx, cx,     cy
        guess = (s1 * 0.7, 0, 1, s0 / 2, s1 / 2)

        # set assume normal plane - no tilt and rotation:
        fn = lambda xy, f, alpha, fx, cx, cy: vignetting((xy[0] * fx, xy[1]), f, alpha,
                                                         cx=cx, cy=cy)

#         mask = fitimg>0.5

        flatfield = fit2dArrayToFn(fitimg, fn, mask=mask,
                                   guess=guess, output_shape=self._orig_shape)[0]

        return flatfield, self.bglevel / self._n, fitimg, mask