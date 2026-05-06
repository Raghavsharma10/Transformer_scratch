def __savorize(self, node: yaml.Node, expected_type: Type) -> yaml.Node:
        """Removes syntactic sugar from the node.

        This calls yatiml_savorize(), first on the class's base \
        classes, then on the class itself.

        Args:
            node: The node to modify.
            expected_type: The type to assume this type is.
        """
        logger.debug('Savorizing node assuming type {}'.format(
            expected_type.__name__))

        for base_class in expected_type.__bases__:
            if base_class in self._registered_classes.values():
                node = self.__savorize(node, base_class)

        if hasattr(expected_type, 'yatiml_savorize'):
            logger.debug('Calling {}.yatiml_savorize()'.format(
                expected_type.__name__))
            cnode = Node(node)
            expected_type.yatiml_savorize(cnode)
            node = cnode.yaml_node
        return node