def get_values(self, attr_name):
        """
        Retrieves the unique set of values seen for the given attribute
        at this node.
        """
        ret = list(self._attr_value_cdist[attr_name].keys()) \
            + list(self._attr_value_counts[attr_name].keys()) \
            + list(self._branches.keys())
        ret = set(ret)
        return ret