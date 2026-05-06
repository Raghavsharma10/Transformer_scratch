def clampings_iter(self, cues=None):
        """
        Iterates over all possible clampings of this experimental setup

        Parameters
        ----------
        cues : Optional[iterable]
            If given, restricts clampings over given species names


        Yields
        ------
        caspo.core.clamping.Clamping
            The next clamping with respect to the experimental setup
        """
        s = cues or list(self.stimuli + self.inhibitors)
        clampings = it.chain.from_iterable(it.combinations(s, r) for r in xrange(len(s) + 1))

        literals_tpl = {}
        for stimulus in self.stimuli:
            literals_tpl[stimulus] = -1

        for c in clampings:
            literals = literals_tpl.copy()
            for cues in c:
                if cues in self.stimuli:
                    literals[cues] = 1
                else:
                    literals[cues] = -1

            yield Clamping(literals.iteritems())