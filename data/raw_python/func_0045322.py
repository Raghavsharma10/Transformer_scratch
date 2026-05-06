def __recognize_user_class(self, node: yaml.Node,
                               expected_type: Type) -> RecResult:
        """Recognize a user-defined class in the node.

        This tries to recognize only exactly the specified class. It \
        recurses down into the class's attributes, but not to its \
        subclasses. See also __recognize_user_classes().

        Args:
            node: The node to recognize.
            expected_type: A user-defined class.

        Returns:
            A list containing the user-defined class, or an empty list.
        """
        logger.debug('Recognizing as a user-defined class')
        loc_str = '{}{}'.format(node.start_mark, os.linesep)
        if hasattr(expected_type, 'yatiml_recognize'):
            try:
                unode = UnknownNode(self, node)
                expected_type.yatiml_recognize(unode)
                return [expected_type], ''
            except RecognitionError as e:
                if len(e.args) > 0:
                    message = ('Error recognizing a {}\n{}because of the'
                               ' following error(s): {}').format(
                                   expected_type.__class__, loc_str,
                                   indent(e.args[0], '    '))
                else:
                    message = 'Error recognizing a {}\n{}'.format(
                        expected_type.__class__, loc_str)
                return [], message
        else:
            if issubclass(expected_type, enum.Enum):
                if (not isinstance(node, yaml.ScalarNode)
                        or node.tag != 'tag:yaml.org,2002:str'):
                    message = 'Expected an enum value from {}\n{}'.format(
                        expected_type.__class__, loc_str)
                    return [], message
            elif (issubclass(expected_type, UserString)
                  or issubclass(expected_type, str)):
                if (not isinstance(node, yaml.ScalarNode)
                        or node.tag != 'tag:yaml.org,2002:str'):
                    message = 'Expected a string matching {}\n{}'.format(
                        expected_type.__class__, loc_str)
                    return [], message
            else:
                # auto-recognize based on constructor signature
                if not isinstance(node, yaml.MappingNode):
                    message = 'Expected a dict/mapping here\n{}'.format(
                        loc_str)
                    return [], message

                for attr_name, type_, required in class_subobjects(
                        expected_type):
                    cnode = Node(node)
                    # try exact match first, dashes if that doesn't match
                    for name in [attr_name, attr_name.replace('_', '-')]:
                        if cnode.has_attribute(name):
                            subnode = cnode.get_attribute(name)
                            recognized_types, message = self.recognize(
                                subnode.yaml_node, type_)
                            if len(recognized_types) == 0:
                                message = ('Failed when checking attribute'
                                           ' {}:\n{}').format(
                                               name, indent(message, '    '))
                                return [], message
                            break
                    else:
                        if required:
                            message = (
                                'Error recognizing a {}\n{}because it'
                                ' is missing an attribute named {}').format(
                                    expected_type.__name__, loc_str, attr_name)
                            if '_' in attr_name:
                                message += ' or maybe {}.\n'.format(
                                    attr_name.replace('_', '-'))
                            else:
                                message += '.\n'
                            return [], message

            return [expected_type], ''