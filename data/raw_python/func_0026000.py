def _getAnalysis(self, axis, analysis, ref=None):
        """
        gets the named analysis on the given axis and caches the result (or reads from the cache if data is available 
        already)
        :param axis: the named axis.
        :param analysis: the analysis name.
        :return: the analysis tuple.
        """
        cache = self.cache.get(str(ref))
        if cache is None:
            cache = {'x': {}, 'y': {}, 'z': {}, 'sum': {}}
            self.cache[str(ref)] = cache
        if axis in cache:
            data = self.cache['raw'].get(axis, None)
            cachedAxis = cache.get(axis)
            if cachedAxis.get(analysis) is None:
                if axis == 'sum':
                    if self._canSum(analysis):
                        fx, Pxx = self._getAnalysis('x', analysis)
                        fy, Pxy = self._getAnalysis('y', analysis)
                        fz, Pxz = self._getAnalysis('z', analysis)
                        # calculate the sum of the squares with an additional weighting for x and y
                        Psum = (((Pxx * 2.2) ** 2) + ((Pxy * 2.4) ** 2) + (Pxz ** 2)) ** 0.5
                        if ref is not None:
                            Psum = librosa.amplitude_to_db(Psum, ref)
                        cachedAxis[analysis] = (fx, Psum)
                    else:
                        return None
                else:
                    cachedAxis[analysis] = getattr(data.highPass(), analysis)(ref=ref)
            return cachedAxis[analysis]
        else:
            return None