def unders_to_dashes_in_keys(self) -> None:
        """Replaces underscores with dashes in key names.

        For each attribute in a mapping, this replaces any underscores \
        in its keys with dashes. Handy because Python does not \
        accept dashes in identifiers, while some YAML-based formats use \
        dashes in their keys.
        """
        for key_node, _ in self.yaml_node.value:
            key_node.value = key_node.value.replace('_', '-')