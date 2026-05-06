def probs(self):
        """
        Returns a list of probabilities for all elements in the form
        [(value1,prob1),(value2,prob2),...].
        """
        return [
            (k, self.counts[k]/float(self.total))
            for k in iterkeys(self.counts)
        ]