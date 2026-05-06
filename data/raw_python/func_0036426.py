def _imp_semsim(self, c1, c2):
        """
        The paper's implicit semantic similarity metric
        involves iteratively computing string overlaps;
        this is a modification where we instead use
        inverse Sift4 distance (a fast approximation of Levenshtein distance).

        Frankly ~ I don't know if this is an appropriate
        substitute, so I'll have to play around with this and see.
        """

        desc1 = self._description(c1)
        desc2 = self._description(c2)

        raw_sim = 1/(sift4(desc1, desc2) + 1)
        return math.log(raw_sim + 1)