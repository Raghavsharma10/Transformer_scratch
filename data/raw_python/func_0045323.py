def __recognize_user_classes(self, node: yaml.Node,
                                 expected_type: Type) -> RecResult:
        """Recognize a user-defined class in the node.

        This returns a list of classes from the inheritance hierarchy \
        headed by expected_type which match the given node and which \
        do not have a registered derived class that matches the given \
        node. So, the returned classes are the most derived matching \
        classes that inherit from expected_type.

        This function recurses down the user's inheritance hierarchy.

        Args:
            node: The node to recognize.
            expected_type: A user-defined class.

        Returns:
            A list containing matched user-defined classes.
        """
        # Let the user override with an explicit tag
        if node.tag in self.__registered_classes:
            return [self.__registered_classes[node.tag]], ''

        recognized_subclasses = []
        message = ''
        for other_class in self.__registered_classes.values():
            if expected_type in other_class.__bases__:
                sub_subclasses, msg = self.__recognize_user_classes(
                    node, other_class)
                recognized_subclasses.extend(sub_subclasses)
                if len(sub_subclasses) == 0:
                    message += msg

        if len(recognized_subclasses) == 0:
            recognized_subclasses, msg = self.__recognize_user_class(
                node, expected_type)
            if len(recognized_subclasses) == 0:
                message += msg

        if len(recognized_subclasses) == 0:
            message = ('Failed to recognize a {}\n{}\nbecause of the following'
                       ' error(s):\n{}').format(expected_type.__name__,
                                                node.start_mark,
                                                indent(msg, '    '))
            return [], message

        if len(recognized_subclasses) > 1:
            message = ('{}{} Could not determine which of the following types'
                       ' this is: {}').format(node.start_mark, os.linesep,
                                              recognized_subclasses)
            return recognized_subclasses, message

        return recognized_subclasses, ''