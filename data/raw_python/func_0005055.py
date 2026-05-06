def phase_diff(self, phase):
        '''Compute the phase differential along a given axis

        Parameters
        ----------
        phase : np.ndarray
            Input phase (in radians)

        Returns
        -------
        dphase : np.ndarray like `phase`
            The phase differential.
        '''

        if self.conv is None:
            axis = 0
        elif self.conv in ('channels_last', 'tf'):
            axis = 0
        elif self.conv in ('channels_first', 'th'):
            axis = 1

        # Compute the phase differential
        dphase = np.empty(phase.shape, dtype=phase.dtype)
        zero_idx = [slice(None)] * phase.ndim
        zero_idx[axis] = slice(1)
        else_idx = [slice(None)] * phase.ndim
        else_idx[axis] = slice(1, None)
        zero_idx = tuple(zero_idx)
        else_idx = tuple(else_idx)
        dphase[zero_idx] = phase[zero_idx]
        dphase[else_idx] = np.diff(np.unwrap(phase, axis=axis), axis=axis)
        return dphase