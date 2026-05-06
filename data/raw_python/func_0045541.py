def dashes_to_unders_in_keys(self) -> None:
        """Replaces dashes with underscores in key names.

        For each attribute in a mapping, this replaces any dashes in \
        its keys with underscores. Handy because Python does not \
        accept dashes in identifiers, while some YAML-based file \
        formats use dashes in their keys.
        """
        for key_node, _ in self.yaml_node.value:
            key_node.value = key_node.value.replace('-', '_')