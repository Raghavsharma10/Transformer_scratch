def _compute_empirical_phi(self, beta):
        """Returns empirical `phi` at the given value of `beta`.

        Does **not** set `phi` attribute, simply returns what
        should be value of `phi` given the current `g` and
        `pi_codon` attributes, plus the passed value of `beta`.
        Note that it uses the passed value of `beta`, **not**
        the current `beta` attribute.

        Initial guess is current value of `phi` attribute."""

        def F(phishort):
            """Difference between `g` and expected `g` given `phishort`."""
            phifull = scipy.append(phishort, 1 - phishort.sum())
            phiprod = scipy.ones(N_CODON, dtype='float')
            for w in range(N_NT):
                phiprod *= phifull[w]**CODON_NT_COUNT[w]
            frx_phiprod = frx * phiprod
            frx_phiprod_codonsum = frx_phiprod.sum(axis=1)
            gexpect = []
            for w in range(N_NT - 1):
                gexpect.append(
                        ((CODON_NT_COUNT[w] * frx_phiprod).sum(axis=1) /
                        frx_phiprod_codonsum).sum() / (3 * self.nsites))
            gexpect = scipy.array(gexpect, dtype='float')
            return self.g[ : -1] - gexpect

        frx = self.pi_codon**beta
        with scipy.errstate(invalid='ignore'):
            result = scipy.optimize.root(F, self.phi[ : -1].copy(),
                    tol=1e-8)
            assert result.success, "Failed: {0}".format(result)
            phishort = result.x
        return scipy.append(phishort, 1 - phishort.sum())