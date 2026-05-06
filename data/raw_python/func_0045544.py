def __attr_index(self, attribute: str) -> Optional[int]:
        """Finds an attribute's index in the yaml_node.value list."""
        attr_index = None
        for i, (key_node, _) in enumerate(self.yaml_node.value):
            if key_node.value == attribute:
                attr_index = i
                break
        return attr_index