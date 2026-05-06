def set_attribute(self, attribute: str,
                      value: Union[ScalarType, yaml.Node]) -> None:
        """Sets the attribute to the given value.

        Use only if is_mapping() returns True.

        If the attribute does not exist, this adds a new attribute, \
        if it does, it will be overwritten.

        If value is a str, int, float, bool or None, the attribute will \
        be set to this value. If you want to set the value to a list or \
        dict containing other values, build a yaml.Node and pass it here.

        Args:
            attribute: Name of the attribute whose value to change.
            value: The value to set.
        """
        start_mark = StreamMark('generated node', 0, 0, 0)
        end_mark = StreamMark('generated node', 0, 0, 0)
        if isinstance(value, str):
            value_node = yaml.ScalarNode('tag:yaml.org,2002:str', value,
                                         start_mark, end_mark)
        elif isinstance(value, bool):
            value_str = 'true' if value else 'false'
            value_node = yaml.ScalarNode('tag:yaml.org,2002:bool', value_str,
                                         start_mark, end_mark)
        elif isinstance(value, int):
            value_node = yaml.ScalarNode('tag:yaml.org,2002:int', str(value),
                                         start_mark, end_mark)
        elif isinstance(value, float):
            value_node = yaml.ScalarNode('tag:yaml.org,2002:float', str(value),
                                         start_mark, end_mark)
        elif value is None:
            value_node = yaml.ScalarNode('tag:yaml.org,2002:null', '',
                                         start_mark, end_mark)
        elif isinstance(value, yaml.Node):
            value_node = value
        else:
            raise TypeError('Invalid kind of value passed to set_attribute()')

        attr_index = self.__attr_index(attribute)
        if attr_index is not None:
            key_node = self.yaml_node.value[attr_index][0]
            self.yaml_node.value[attr_index] = key_node, value_node
        else:
            key_node = yaml.ScalarNode('tag:yaml.org,2002:str', attribute,
                                       start_mark, end_mark)
            self.yaml_node.value.append((key_node, value_node))