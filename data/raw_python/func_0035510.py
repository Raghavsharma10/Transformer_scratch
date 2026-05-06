def spielman_wr(self, norm=True):
        """Returns a list of site-specific omega values calculated from the `ExpCM`.

            Args:
                `norm` (bool)
                    If `True`, normalize the `omega_r` values by the ExpCM
                    gene-wide `omega`.

            Returns:
                `wr` (list)
                    list of `omega_r` values of length `nsites`

        Following
        `Spielman and Wilke, MBE, 32:1097-1108 <https://doi.org/10.1093/molbev/msv003>`_,
        we can predict the `dN/dS` value for each site `r`,
        :math:`\\rm{spielman}\\omega_r`, from the `ExpCM`.

        When `norm` is `False`, the `omega_r` values are defined as
        :math:`\\rm{spielman}\\omega_r = \\frac{\\sum_x \\sum_{y \\in N_x}p_{r,x}\
        P_{r,xy}}{\\sum_x \\sum_{y \\in Nx}p_{r,x}Q_{xy}}`,
        where `r,x,y`, :math:`p_{r,x}`, :math:`P_{r,xy}`, and :math:`Q_{x,y}`
        have the same definitions as in the main `ExpCM` doc string and :math:`N_{x}`
        is the set of codons which are non-synonymous to codon `x` and differ from
        `x` by one nucleotide.

        When `norm` is `True`, the `omega_r` values above are divided by the
        ExpCM `omega` value."""

        wr = []
        for r in range(self.nsites):
            num = 0
            den = 0
            for i in range(N_CODON):
                j = scipy.intersect1d(scipy.where(CODON_SINGLEMUT[i]==True)[0],
                        scipy.where(CODON_NONSYN[i]==True)[0])
                p_i = self.stationarystate[r][i]
                P_xy = self.Prxy[r][i][j].sum()
                if norm:
                    P_xy = P_xy/self.omega
                Q_xy = self.Qxy[i][j].sum()
                num += (p_i * P_xy)
                den += (p_i * Q_xy)
            result = num/den
            wr.append(result)
        return wr