def set_value(self, value: ScalarType) -> None:
        """Sets the value of the node to a scalar value.

        After this, is_scalar(type(value)) will return true.

        Args:
            value: The value to set this node to, a str, int, float, \
                    bool, or None.
        """
        if isinstance(value, bool):
            value_str = 'true' if value else 'false'
        else:
            value_str = str(value)
        start_mark = self.yaml_node.start_mark
        end_mark = self.yaml_node.end_mark
        # If we're of a class type, then we want to keep that tag so that the
        # correct Constructor is called. If we're a built-in type, set the tag
        # to the appropriate YAML tag.
        tag = self.yaml_node.tag
        if tag.startswith('tag:yaml.org,2002:'):
            tag = scalar_type_to_tag[type(value)]
        new_node = yaml.ScalarNode(tag, value_str, start_mark, end_mark)
        self.yaml_node = new_node