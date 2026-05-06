def _get_attribute_value_for_node(self, record):
        """
        Gets the closest value for the current node's attribute matching the
        given record.
        """
        
        # Abort if this node has not get split on an attribute. 
        if self.attr_name is None:
            return
        
        # Otherwise, lookup the attribute value for this node in the
        # given record.
        attr = self.attr_name
        attr_value = record[attr]
        attr_values = self.get_values(attr)
        if attr_value in attr_values:
            return attr_value
        else:
            # The value of the attribute in the given record does not directly
            # map to any previously known values, so apply a missing value
            # policy.
            policy = self.tree.missing_value_policy.get(attr)
            assert policy, \
                ("No missing value policy specified for attribute %s.") \
                % (attr,)
            if policy == USE_NEAREST:
                # Use the value that the tree has seen that's also has the
                # smallest Euclidean distance to the actual value.
                assert self.tree.data.header_types[attr] \
                    in (ATTR_TYPE_DISCRETE, ATTR_TYPE_CONTINUOUS), \
                    "The use-nearest policy is invalid for nominal types."
                nearest = (1e999999, None)
                for _value in attr_values:
                    nearest = min(
                        nearest,
                        (abs(_value - attr_value), _value))
                _, nearest_value = nearest
                return nearest_value
            else:
                raise Exception("Unknown missing value policy: %s" % (policy,))