def get_total_term_frequency(self, term):
        """
        Gets the frequency of the specified term in the entire corpus
        added to the HashedIndex.
        """
        if term not in self._terms:
            raise IndexError(TERM_DOES_NOT_EXIST)

        return sum(self._terms[term].values())