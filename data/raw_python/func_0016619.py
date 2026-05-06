def get_best_splitting_attr(self):
        """
        Returns the name of the attribute with the highest gain.
        """
        best = (-1e999999, None)
        for attr in self.attributes:
            best = max(best, (self.get_gain(attr), attr))
        best_gain, best_attr = best
        return best_attr