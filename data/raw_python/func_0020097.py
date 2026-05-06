def update_gp(self):
        '''
        Calls :py:func:`gp.GetKernelParams` to optimize the GP and obtain the
        covariance matrix for the regression.

        '''

        self.kernel_params = GetKernelParams(self.time, self.flux,
                                             self.fraw_err,
                                             mask=self.mask,
                                             guess=self.kernel_params,
                                             kernel=self.kernel,
                                             giter=self.giter,
                                             gmaxf=self.gmaxf)