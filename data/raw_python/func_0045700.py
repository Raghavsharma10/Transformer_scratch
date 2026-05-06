def __process_node(self, node: yaml.Node,
                       expected_type: Type) -> yaml.Node:
        """Processes a node.

        This is the main function that implements yatiml's \
        functionality. It figures out how to interpret this node \
        (recognition), then applies syntactic sugar, and finally \
        recurses to the subnodes, if any.

        Args:
            node: The node to process.
            expected_type: The type we expect this node to be.

        Returns:
            The transformed node, or a transformed copy.
        """
        logger.info('Processing node {} expecting type {}'.format(
            node, expected_type))

        # figure out how to interpret this node
        recognized_types, message = self.__recognizer.recognize(
            node, expected_type)

        if len(recognized_types) != 1:
            raise RecognitionError(message)

        recognized_type = recognized_types[0]

        # remove syntactic sugar
        logger.debug('Savorizing node {}'.format(node))
        if recognized_type in self._registered_classes.values():
            node = self.__savorize(node, recognized_type)
        logger.debug('Savorized, now {}'.format(node))

        # process subnodes
        logger.debug('Recursing into subnodes')
        if is_generic_list(recognized_type):
            if node.tag != 'tag:yaml.org,2002:seq':
                raise RecognitionError('{}{}Expected a {} here'.format(
                    node.start_mark, os.linesep,
                    type_to_desc(expected_type)))
            for item in node.value:
                self.__process_node(item,
                                    generic_type_args(recognized_type)[0])
        elif is_generic_dict(recognized_type):
            if node.tag != 'tag:yaml.org,2002:map':
                raise RecognitionError('{}{}Expected a {} here'.format(
                    node.start_mark, os.linesep,
                    type_to_desc(expected_type)))
            for _, value_node in node.value:
                self.__process_node(value_node,
                                    generic_type_args(recognized_type)[1])

        elif recognized_type in self._registered_classes.values():
            if (not issubclass(recognized_type, enum.Enum)
                    and not issubclass(recognized_type, str)
                    and not issubclass(recognized_type, UserString)):
                for attr_name, type_, _ in class_subobjects(recognized_type):
                    cnode = Node(node)
                    if cnode.has_attribute(attr_name):
                        subnode = cnode.get_attribute(attr_name)
                        new_subnode = self.__process_node(
                            subnode.yaml_node, type_)
                        cnode.set_attribute(attr_name, new_subnode)
        else:
            logger.debug('Not a generic class or a user-defined class, not'
                         ' recursing')

        node.tag = self.__type_to_tag(recognized_type)
        logger.debug('Finished processing node {}'.format(node))
        return node