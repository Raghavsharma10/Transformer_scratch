def get_outliers(self):
        '''
        Performs iterative sigma clipping to get outliers.

        '''

        log.info("Clipping outliers...")
        log.info('Iter %d/%d: %d outliers' %
                 (0, self.oiter, len(self.outmask)))

        def M(x): return np.delete(x, np.concatenate(
            [self.nanmask, self.badmask, self.transitmask]), axis=0)
        t = M(self.time)
        outmask = [np.array([-1]), np.array(self.outmask)]

        # Loop as long as the last two outlier arrays aren't equal
        while not np.array_equal(outmask[-2], outmask[-1]):

            # Check if we've done this too many times
            if len(outmask) - 1 > self.oiter:
                log.error('Maximum number of iterations in ' +
                          '``get_outliers()`` exceeded. Skipping...')
                break

            # Check if we're going in circles
            if np.any([np.array_equal(outmask[-1], i) for i in outmask[:-1]]):
                log.error('Function ``get_outliers()`` ' +
                          'is going in circles. Skipping...')
                break

            # Compute the model to get the flux
            self.compute()

            # Get the outliers
            f = SavGol(M(self.flux))
            med = np.nanmedian(f)
            MAD = 1.4826 * np.nanmedian(np.abs(f - med))
            inds = np.where((f > med + self.osigma * MAD) |
                            (f < med - self.osigma * MAD))[0]

            # Project onto unmasked time array
            inds = np.array([np.argmax(self.time == t[i]) for i in inds])
            self.outmask = np.array(inds, dtype=int)

            # Add them to the running list
            outmask.append(np.array(inds))

            # Log
            log.info('Iter %d/%d: %d outliers' %
                     (len(outmask) - 2, self.oiter, len(self.outmask)))