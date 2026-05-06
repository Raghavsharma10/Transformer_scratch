def rename_attribute(self, attribute: str, new_name: str) -> None:
        """Renames an attribute.

        Use only if is_mapping() returns true.

        If the attribute does not exist, this will do nothing.

        Args:
            attribute: The (old) name of the attribute to rename.
            new_name: The new name to rename it to.
        """
        for key_node, _ in self.yaml_node.value:
            if key_node.value == attribute:
                key_node.value = new_name
                break