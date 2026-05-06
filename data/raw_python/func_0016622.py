def get_value_ddist(self, attr_name, attr_value):
        """
        Returns the class value probability distribution of the given
        attribute value.
        """
        assert not self.tree.data.is_continuous_class, \
            "Discrete distributions are only maintained for " + \
            "discrete class types."
        ddist = DDist()
        cls_counts = self._attr_class_value_counts[attr_name][attr_value]
        for cls_value, cls_count in iteritems(cls_counts):
            ddist.add(cls_value, count=cls_count)
        return ddist