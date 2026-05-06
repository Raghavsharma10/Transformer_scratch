def get_attribute(self, attribute: str) -> 'Node':
        """Returns the node representing the given attribute's value.

        Use only if is_mapping() returns true.

        Args:
            attribute: The name of the attribute to retrieve.

        Raises:
            KeyError: If the attribute does not exist.

        Returns:
            A node representing the value.
        """
        matches = [
            value_node for key_node, value_node in self.yaml_node.value
            if key_node.value == attribute
        ]
        if len(matches) != 1:
            raise SeasoningError(
                'Attribute not found, or found multiple times: {}'.format(
                    matches))
        return Node(matches[0])