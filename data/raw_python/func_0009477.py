def error(self, nCells=15):
        '''
        calculate the standard deviation of all fitted images, 
        averaged to a grid
        '''
        s0, s1 = self.fits[0].shape
        aR = s0 / s1
        if aR > 1:
            ss0 = int(nCells)
            ss1 = int(ss0 / aR)
        else:
            ss1 = int(nCells)
            ss0 = int(ss1 * aR)
        L = len(self.fits)

        arr = np.array(self.fits)
        arr[np.array(self._fit_masks)] = np.nan
        avg = np.tile(np.nanmean(arr, axis=0), (L, 1, 1))
        arr = (arr - avg) / avg

        out = np.empty(shape=(L, ss0, ss1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            for n, f in enumerate(arr):
                out[n] = subCell2DFnArray(f, np.nanmean, (ss0, ss1))

        return np.nanmean(out**2)**0.5