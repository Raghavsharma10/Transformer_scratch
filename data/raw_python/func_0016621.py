def get_gain(self, attr_name):
        """
        Calculates the information gain from splitting on the given attribute.
        """
        subset_entropy = 0.0
        for value in iterkeys(self._attr_value_counts[attr_name]):
            value_prob = self.get_value_prob(attr_name, value)
            e = self.get_entropy(attr_name, value)
            subset_entropy += value_prob * e
        return (self.main_entropy - subset_entropy)