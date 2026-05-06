def optimize(self, piter=3, pmaxf=300, ppert=0.1):
        '''
        Runs :py:obj:`pPLD` on the target in an attempt to further optimize the
        values of the PLD priors. See :py:class:`everest.detrender.pPLD`.

        '''

        self._save_npz()
        optimized = pPLD(self.ID, piter=piter, pmaxf=pmaxf,
                         ppert=ppert, debug=True, clobber=True)
        optimized.publish()
        self.reset()