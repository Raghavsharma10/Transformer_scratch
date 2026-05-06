def require_attribute_value(
            self, attribute: str,
            value: Union[int, str, float, bool, None]) -> None:
        """Require an attribute on the node to have a particular value.

        This requires the attribute to exist, and to have the given value \
        and corresponding type. Handy for in your yatiml_recognize() \
        function.

        Args:
            attribute: The name of the attribute / mapping key.
            value: The value the attribute must have to recognize an \
                    object of this type.

        Raises:
            RecognitionError: If the attribute does not exist, or does \
                    not have the required value.
        """
        found = False
        for key_node, value_node in self.yaml_node.value:
            if (key_node.tag == 'tag:yaml.org,2002:str'
                    and key_node.value == attribute):
                found = True
                node = Node(value_node)
                if not node.is_scalar(type(value)):
                    raise RecognitionError(
                            ('{}{}Incorrect attribute type where value {}'
                             ' of type {} was required').format(
                                self.yaml_node.start_mark, os.linesep,
                                value, type(value)))
                if node.get_value() != value:
                    raise RecognitionError(
                        ('{}{}Incorrect attribute value'
                         ' {} where {} was required').format(
                             self.yaml_node.start_mark, os.linesep,
                             value_node.value, value))

        if not found:
            raise RecognitionError(
                ('{}{}Required attribute {} not found').format(
                    self.yaml_node.start_mark, os.linesep, attribute))