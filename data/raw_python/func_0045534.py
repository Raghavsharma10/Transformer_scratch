def has_attribute(self, attribute: str) -> bool:
        """Whether the node has an attribute with the given name.

        Use only if is_mapping() returns True.

        Args:
            attribute: The name of the attribute to check for.

        Returns:
            True iff the attribute is present.
        """
        return any([
            key_node.value == attribute for key_node, _ in self.yaml_node.value
        ])