def get_value_prob(self, attr_name, value):
        """
        Returns the value probability of the given attribute at this node.
        """
        if attr_name not in self._attr_value_count_totals:
            return
        n = self._attr_value_counts[attr_name][value]
        d = self._attr_value_count_totals[attr_name]
        return n/float(d)