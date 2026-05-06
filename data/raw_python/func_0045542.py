def seq_attribute_to_map(self,
                             attribute: str,
                             key_attribute: str,
                             value_attribute: Optional[str] = None,
                             strict: Optional[bool] = True) -> None:
        """Converts a sequence attribute to a map.

        This function takes an attribute of this Node that is \
        a sequence of mappings and turns it into a mapping of mappings. \
        It assumes that each of the mappings in the original sequence \
        has an attribute containing a unique value, which it will use \
        as a key for the new outer mapping.

        An example probably helps. If you have a Node representing \
        this piece of YAML::

            items:
            - item_id: item1
              description: Basic widget
              price: 100.0
            - item_id: item2
              description: Premium quality widget
              price: 200.0

        and call seq_attribute_to_map('items', 'item_id'), then the \
        Node will be modified to represent this::

            items:
              item1:
                description: Basic widget
                price: 100.0
              item2:
                description: Premium quality widget
                price: 200.0

        which is often more intuitive for people to read and write.

        If the attribute does not exist, or is not a sequence of \
        mappings, this function will silently do nothing. If the keys \
        are not unique and strict is False, it will also do nothing. If \
        the keys are not unique and strict is True, it will raise an \
        error.

        With thanks to the makers of the Common Workflow Language for \
        the idea.

        Args:
            attribute: Name of the attribute whose value to modify.
            key_attribute: Name of the attribute in each item to use \
                    as a key for the new mapping.
            strict: Whether to give an error if the intended keys are \
                    not unique.

        Raises:
            SeasoningError: If the keys are not unique and strict is \
                    True.
        """
        if not self.has_attribute(attribute):
            return

        attr_node = self.get_attribute(attribute)
        if not attr_node.is_sequence():
            return

        start_mark = attr_node.yaml_node.start_mark
        end_mark = attr_node.yaml_node.end_mark

        # check that all list items are mappings and that the keys are unique
        # strings
        seen_keys = set()  # type: Set[str]
        for item in attr_node.seq_items():
            key_attr_node = item.get_attribute(key_attribute)
            if not key_attr_node.is_scalar(str):
                raise SeasoningError(
                    ('Attribute names must be strings in'
                     'YAtiML, {} is not a string.').format(key_attr_node))
            if key_attr_node.get_value() in seen_keys:
                if strict:
                    raise SeasoningError(
                        ('Found a duplicate key {}: {} when'
                         ' converting from sequence to mapping'.format(
                             key_attribute, key_attr_node.get_value())))
                return
            seen_keys.add(key_attr_node.get_value())  # type: ignore

        # construct mapping
        mapping_values = list()
        for item in attr_node.seq_items():
            # we've already checked that it's a SequenceNode above
            key_node = item.get_attribute(key_attribute).yaml_node
            item.remove_attribute(key_attribute)
            if value_attribute is not None:
                value_node = item.get_attribute(value_attribute).yaml_node
                if len(item.yaml_node.value) == 1:
                    # no other attributes, use short form
                    mapping_values.append((key_node, value_node))
                else:
                    mapping_values.append((key_node, item.yaml_node))
            else:
                mapping_values.append((key_node, item.yaml_node))

        # create mapping node
        mapping = yaml.MappingNode('tag:yaml.org,2002:map', mapping_values,
                                   start_mark, end_mark)
        self.set_attribute(attribute, mapping)