def _update_Prxy_diag(self):
        """Update `D`, `A`, `Ainv` from `Prxy`, `prx`."""
        for r in range(self.nsites):
            pr_half = self.prx[r]**0.5
            pr_neghalf = self.prx[r]**-0.5
            #symm_pr = scipy.dot(scipy.diag(pr_half), scipy.dot(self.Prxy[r], scipy.diag(pr_neghalf)))
            symm_pr = (pr_half * (self.Prxy[r] * pr_neghalf).transpose()).transpose()
            # assert scipy.allclose(symm_pr, symm_pr.transpose())
            (evals, evecs) = scipy.linalg.eigh(symm_pr)
            # assert scipy.allclose(scipy.linalg.inv(evecs), evecs.transpose())
            # assert scipy.allclose(symm_pr, scipy.dot(evecs, scipy.dot(scipy.diag(evals), evecs.transpose())))
            self.D[r] = evals
            self.Ainv[r] = evecs.transpose() * pr_half
            self.A[r] = (pr_neghalf * evecs.transpose()).transpose()