def has_attribute_type(self, attribute: str, typ: Type) -> bool:
        """Whether the given attribute exists and has a compatible type.

        Returns true iff the attribute exists and is an instance of \
        the given type. Matching between types passed as typ and \
        yaml node types is as follows:

        +---------+-------------------------------------------+
        |   typ   |                 yaml                      |
        +=========+===========================================+
        |   str   |      ScalarNode containing string         |
        +---------+-------------------------------------------+
        |   int   |      ScalarNode containing int            |
        +---------+-------------------------------------------+
        |  float  |      ScalarNode containing float          |
        +---------+-------------------------------------------+
        |   bool  |      ScalarNode containing bool           |
        +---------+-------------------------------------------+
        |   None  |      ScalarNode containing null           |
        +---------+-------------------------------------------+
        |   list  |      SequenceNode                         |
        +---------+-------------------------------------------+
        |   dict  |      MappingNode                          |
        +---------+-------------------------------------------+

        Args:
            attribute: The name of the attribute to check.
            typ: The type to check against.

        Returns:
            True iff the attribute exists and matches the type.
        """
        if not self.has_attribute(attribute):
            return False

        attr_node = self.get_attribute(attribute).yaml_node

        if typ in scalar_type_to_tag:
            tag = scalar_type_to_tag[typ]
            return attr_node.tag == tag
        elif typ == list:
            return isinstance(attr_node, yaml.SequenceNode)
        elif typ == dict:
            return isinstance(attr_node, yaml.MappingNode)

        raise ValueError('Invalid argument for typ attribute')