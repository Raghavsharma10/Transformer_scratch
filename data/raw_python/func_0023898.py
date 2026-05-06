def _compute_r2l2(self, tau, return_l=False):
        r"""Compute the anisotropic :math:`r^2/l^2` term for the given `tau`.
        
        Here, :math:`\tau=X_i-X_j` is the difference vector. Computes
        .. math::
            
            \frac{r^2}{l^2} = \sum_i\frac{\tau_i^2}{l_{i}^{2}}
        
        Assumes that the length parameters are the last `num_dim` elements of
        :py:attr:`self.params`.
        
        Where `l` and `tau` are both zero, that term is set to zero.
        
        Parameters
        ----------
        tau : :py:class:`Array`, (`M`, `D`)
            `M` inputs with dimension `D`.
        return_l : bool, optional
            Set to True to return a tuple of (`tau`, `l_mat`), where `l_mat`
            is the matrix of length scales to match the shape of `tau`. Default
            is False (only return `tau`).
        
        Returns
        -------
        r2l2 : :py:class:`Array`, (`M`,)
            Anisotropically scaled distances squared.
        l_mat : :py:class:`Array`, (`M`, `D`)
            The (`D`,) array of length scales repeated for each of the `M`
            inputs. Only returned if `return_l` is True.
        """
        l_mat = scipy.tile(self.params[-self.num_dim:], (tau.shape[0], 1))
        tau_over_l = tau / l_mat
        tau_over_l[(tau == 0) & (l_mat == 0)] = 0.0
        r2l2 = scipy.sum((tau_over_l)**2, axis=1)
        if return_l:
            return (r2l2, l_mat)
        else:
            return r2l2