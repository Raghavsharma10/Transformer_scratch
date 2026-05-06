def setup(self, **kwargs):
        '''
        This is called during production de-trending, prior to
        calling the :py:obj:`Detrender.run()` method.

        :param inter piter: The number of iterations in the minimizer. \
               Default 3
        :param int pmaxf: The maximum number of function evaluations per \
               iteration. Default 300
        :param float ppert: The fractional amplitude of the perturbation on \
               the initial guess. Default 0.1

        '''

        # Check for saved model
        clobber = self.clobber
        self.clobber = False
        if not self.load_model('nPLD'):
            raise Exception("Can't find `nPLD` model for target.")
        self.clobber = clobber

        # Powell iterations
        self.piter = kwargs.get('piter', 3)
        self.pmaxf = kwargs.get('pmaxf', 300)
        self.ppert = kwargs.get('ppert', 0.1)