def setup(self, **kwargs):
        '''
        This is called during production de-trending, prior to
        calling the :py:obj:`Detrender.run()` method.

        :param str parent_model: The name of the model to operate on. \
               Default `nPLD`

        '''

        # Load the parent model
        self.parent_model = kwargs.get('parent_model', 'nPLD')
        if not self.load_model(self.parent_model):
            raise Exception('Unable to load parent model.')

        # Save static copies of the de-trended flux,
        # the outlier mask and the lambda array
        self._norm = np.array(self.flux)
        self.recmask = np.array(self.mask)
        self.reclam = np.array(self.lam)

        # Now reset the model params
        self.optimize_gp = False
        nseg = len(self.breakpoints)
        self.lam_idx = -1
        self.lam = [
            [1e5] + [None for i in range(self.pld_order - 1)]
            for b in range(nseg)]
        self.cdpp_arr = np.array([np.nan for b in range(nseg)])
        self.cdppr_arr = np.array([np.nan for b in range(nseg)])
        self.cdppv_arr = np.array([np.nan for b in range(nseg)])
        self.cdpp = np.nan
        self.cdppr = np.nan
        self.cdppv = np.nan
        self.cdppg = np.nan
        self.model = np.zeros_like(self.time)
        self.loaded = True