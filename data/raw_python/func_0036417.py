def last_n_results(self, n):
        """
        Helper method for returning a set number of the previous check results.
        """
        return list(
            itertools.islice(
                self.results, len(self.results) - n, len(self.results)
            )
        )