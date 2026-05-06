def turningpoint(self):
        """Turning point (index and value tuple) in the recession part of the
        MA approximation of the instantaneous unit hydrograph."""
        coefs = self.coefs
        old_dc = coefs[1]-coefs[0]
        for idx in range(self.order-2):
            new_dc = coefs[idx+2]-coefs[idx+1]
            if (old_dc < 0.) and (new_dc > old_dc):
                return idx, coefs[idx]
            old_dc = new_dc
        raise RuntimeError(
            'Not able to detect a turning point in the impulse response '
            'defined by the MA coefficients %s.'
            % objecttools.repr_values(coefs))