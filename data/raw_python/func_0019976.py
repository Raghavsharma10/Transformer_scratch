def percentile(self, val):
        """
        Get the distribution value at a given percentile or set of percentiles.
        This follows the NIST method for calculating percentiles.
        
        Parameters
        ----------
        val : scalar or array
            Either a single value or an array of values between 0 and 1.
        
        Returns
        -------
        out : scalar or array
            The actual distribution value that appears at the requested
            percentile value or values
            
        """
        try:
            # test to see if an input is given as an array
            out = [self.percentile(vi) for vi in val]
        except (ValueError, TypeError):
            if val <= 0:
                out = float(min(self._mcpts))
            elif val >= 1:
                out = float(max(self._mcpts))
            else:
                tmp = np.sort(self._mcpts)
                n = val * (len(tmp) + 1)
                k, d = int(n), n - int(n)
                out = float(tmp[k] + d * (tmp[k + 1] - tmp[k]))
        if isinstance(val, np.ndarray):
            out = np.array(out)
        return out