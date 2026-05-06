def set_leaf_dist(self, attr_value, dist):
        """
        Sets the probability distribution at a leaf node.
        """
        assert self.attr_name
        assert self.tree.data.is_valid(self.attr_name, attr_value), \
            "Value %s is invalid for attribute %s." \
                % (attr_value, self.attr_name)
        if self.is_continuous_class:
            assert isinstance(dist, CDist)
            assert self.attr_name
            self._attr_value_cdist[self.attr_name][attr_value] = dist.copy()
#            self.n += dist.count
        else:
            assert isinstance(dist, DDist)
            # {attr_name:{attr_value:count}}
            self._attr_value_counts[self.attr_name][attr_value] += 1
            # {attr_name:total}
            self._attr_value_count_totals[self.attr_name] += 1
            # {attr_name:{attr_value:{class_value:count}}}
            for cls_value, cls_count in iteritems(dist.counts):
                self._attr_class_value_counts[self.attr_name][attr_value] \
                    [cls_value] += cls_count