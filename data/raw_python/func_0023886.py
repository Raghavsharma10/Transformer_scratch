def _compute_y_wrapper(self, *args):
        r"""Convert tau to :math:`y=\sqrt{2\nu\sum_i(\tau_i^2/l_i^2)}`.
        
        Takes `tau` as an argument list for compatibility with :py:func:`mpmath.diff`.
        
        Parameters
        ----------
        tau[0] : scalar float
            First element of `tau`.
        tau[1] : And so on...
        
        Returns
        -------
        y : scalar float
            Inner part of Matern kernel at the given `tau`.
        """
        return self._compute_y(scipy.atleast_2d(scipy.asarray(args, dtype=float)))