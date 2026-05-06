def get_val(self, x):
        """Convert cat value into num between 0 and 1."""
        interval, mean, std = self.probability_map[x]
        new_val = norm.rvs(mean, std)
        return new_val