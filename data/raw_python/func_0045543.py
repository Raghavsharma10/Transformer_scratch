def map_attribute_to_seq(self,
                             attribute: str,
                             key_attribute: str,
                             value_attribute: Optional[str] = None) -> None:
        """Converts a mapping attribute to a sequence.

        This function takes an attribute of this Node whose value \
        is a mapping or a mapping of mappings and turns it into a \
        sequence of mappings. Each entry in the original mapping is \
        converted to an entry in the list. If only a key attribute is \
        given, then each entry in the original mapping must map to a \
        (sub)mapping. This submapping becomes the corresponding list \
        entry, with the key added to it as an additional attribute. If a \
        value attribute is also given, then an entry in the original \
        mapping may map to any object. If the mapped-to object is a \
        mapping, the conversion is as before, otherwise a new \
        submapping is created, and key and value are added using the \
        given key and value attribute names.

        An example probably helps. If you have a Node representing \
        this piece of YAML::

            items:
              item1:
                description: Basic widget
                price: 100.0
              item2:
                description: Premium quality widget
                price: 200.0

        and call map_attribute_to_seq('items', 'item_id'), then the \
        Node will be modified to represent this::

            items:
            - item_id: item1
              description: Basic widget
              price: 100.0
            - item_id: item2
              description: Premium quality widget
              price: 200.0

        which once converted to an object is often easier to deal with \
        in code.

        Slightly more complicated, this YAML::

            items:
              item1: Basic widget
              item2:
                description: Premium quality widget
                price: 200.0

        when passed through map_attribute_to_seq('items', 'item_id', \
        'description'), will result in th equivalent of::

            items:
            - item_id: item1
              description: Basic widget
            - item_id: item2
              description: Premium quality widget
              price: 200.0

        If the attribute does not exist, or is not a mapping, this \
        function will silently do nothing.

        With thanks to the makers of the Common Workflow Language for \
        the idea.

        Args:
            attribute: Name of the attribute whose value to modify.
            key_attribute: Name of the new attribute in each item to \
                    add with the value of the key.
            value_attribute: Name of the new attribute in each item to \
                    add with the value of the key.
        """
        if not self.has_attribute(attribute):
            return

        attr_node = self.get_attribute(attribute)
        if not attr_node.is_mapping():
            return

        start_mark = attr_node.yaml_node.start_mark
        end_mark = attr_node.yaml_node.end_mark
        object_list = []
        for item_key, item_value in attr_node.yaml_node.value:
            item_value_node = Node(item_value)
            if not item_value_node.is_mapping():
                if value_attribute is None:
                    return
                ynode = item_value_node.yaml_node
                item_value_node.make_mapping()
                item_value_node.set_attribute(value_attribute, ynode)

            item_value_node.set_attribute(key_attribute, item_key.value)
            object_list.append(item_value_node.yaml_node)
        seq_node = yaml.SequenceNode('tag:yaml.org,2002:seq', object_list,
                                     start_mark, end_mark)
        self.set_attribute(attribute, seq_node)