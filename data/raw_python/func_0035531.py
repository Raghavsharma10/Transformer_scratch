def _update_Pxy_diag(self):
        """Update `D`, `A`, `Ainv` from `Pxy`, `Phi_x`."""
        for r in range(1):
            Phi_x_half = self.Phi_x**0.5
            Phi_x_neghalf = self.Phi_x**-0.5
            #symm_p = scipy.dot(scipy.diag(Phi_x_half), scipy.dot(self.Pxy[r], scipy.diag(Phi_x_neghalf)))
            symm_p = (Phi_x_half * (self.Pxy[r] * Phi_x_neghalf).transpose()).transpose()
            #assert scipy.allclose(symm_p, symm_p.transpose())
            (evals, evecs) = scipy.linalg.eigh(symm_p)
            #assert scipy.allclose(scipy.linalg.inv(evecs), evecs.transpose())
            #assert scipy.allclose(symm_pr, scipy.dot(evecs, scipy.dot(scipy.diag(evals), evecs.transpose())))
            self.D[r] = evals
            self.Ainv[r] = evecs.transpose() * Phi_x_half
            self.A[r] = (Phi_x_neghalf * evecs.transpose()).transpose()