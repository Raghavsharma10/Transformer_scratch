def remove_attribute(self, attribute: str) -> None:
        """Remove an attribute from the node.

        Use only if is_mapping() returns True.

        Args:
            attribute: The name of the attribute to remove.
        """
        attr_index = self.__attr_index(attribute)
        if attr_index is not None:
            self.yaml_node.value.pop(attr_index)