def set_missing_value_policy(self, policy, target_attr_name=None):
        """
        Sets the behavior for one or all attributes to use when traversing the
        tree using a query vector and it encounters a branch that does not
        exist.
        """
        assert policy in MISSING_VALUE_POLICIES, \
            "Unknown policy: %s" % (policy,)
        for attr_name in self.data.attribute_names:
            if target_attr_name is not None and target_attr_name != attr_name:
                continue
            self.missing_value_policy[attr_name] = policy