def __strip_tags(self, node: yaml.Node) -> None:
        """Strips tags from mappings in the tree headed by node.

        This keeps yaml from constructing any objects in this tree.

        Args:
            node: Head of the tree to strip
        """
        if isinstance(node, yaml.SequenceNode):
            for subnode in node.value:
                self.__strip_tags(subnode)
        elif isinstance(node, yaml.MappingNode):
            node.tag = 'tag:yaml.org,2002:map'
            for key_node, value_node in node.value:
                self.__strip_tags(key_node)
                self.__strip_tags(value_node)