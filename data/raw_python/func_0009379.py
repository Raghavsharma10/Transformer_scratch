def permute(self, permutations=99, alternative='two.sided'):
        """
        Generate ransom spatial permutations for inference on LISA vectors.

        Parameters
        ----------
        permutations : int, optional
            Number of random permutations of observations.
        alternative : string, optional
            Type of alternative to form in generating p-values.
            Options are: `two-sided` which tests for difference between observed
            counts and those obtained from the permutation distribution;
            `positive` which tests the alternative that the focal unit and its
            lag move in the same direction over time; `negative` which tests
            that the focal unit and its lag move in opposite directions over
            the interval.
        """
        rY = self.Y.copy()
        idxs = np.arange(len(rY))
        counts = np.zeros((permutations, len(self.counts)))
        for m in range(permutations):
            np.random.shuffle(idxs)
            res = self._calc(rY[idxs, :], self.w, self.k)
            counts[m] = res['counts']
        self.counts_perm = counts
        self.larger_perm = np.array(
            [(counts[:, i] >= self.counts[i]).sum() for i in range(self.k)])
        self.smaller_perm = np.array(
            [(counts[:, i] <= self.counts[i]).sum() for i in range(self.k)])
        self.expected_perm = counts.mean(axis=0)
        self.alternative = alternative

        # pvalue logic
        # if P is the proportion that are as large for a one sided test (larger
        # than), then
        # p=P.
        #
        # For a two-tailed test, if P < .5, p = 2 * P, else, p = 2(1-P)
        # Source: Rayner, J. C. W., O. Thas, and D. J. Best. 2009. "Appendix B:
        # Parametric Bootstrap P-Values." In Smooth Tests of Goodness of Fit,
        # 247. John Wiley and Sons.
        # Note that the larger and smaller counts would be complements (except
        # for the shared equality, for
        # a given bin in the circular histogram. So we only need one of them.

        # We report two-sided p-values for each bin as the default
        # since a priori there could # be different alternatives for each bin
        # depending on the problem at hand.

        alt = alternative.upper()
        if alt == 'TWO.SIDED':
            P = (self.larger_perm + 1) / (permutations + 1.)
            mask = P < 0.5
            self.p = mask * 2 * P + (1 - mask) * 2 * (1 - P)
        elif alt == 'POSITIVE':
            # NE, SW sectors are higher, NW, SE are lower
            POS = _POS8
            if self.k == 4:
                POS = _POS4
            L = (self.larger_perm + 1) / (permutations + 1.)
            S = (self.smaller_perm + 1) / (permutations + 1.)
            P = POS * L + (1 - POS) * S
            self.p = P
        elif alt == 'NEGATIVE':
            # NE, SW sectors are lower, NW, SE are higher
            NEG = _NEG8
            if self.k == 4:
                NEG = _NEG4
            L = (self.larger_perm + 1) / (permutations + 1.)
            S = (self.smaller_perm + 1) / (permutations + 1.)
            P = NEG * L + (1 - NEG) * S
            self.p = P
        else:
            print(('Bad option for alternative: %s.' % alternative))