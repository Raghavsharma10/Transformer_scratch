def require_scalar(self, *args: Type) -> None:
        """Require the node to be a scalar.

        If additional arguments are passed, these are taken as a list \
        of valid types; if the node matches one of these, then it is \
        accepted.

        Example:
            # Match either an int or a string
            node.require_scalar(int, str)

        Arguments:
            args: One or more types to match one of.
        """
        node = Node(self.yaml_node)
        if len(args) == 0:
            if not node.is_scalar():
                raise RecognitionError(('{}{}A scalar is required').format(
                    self.yaml_node.start_mark, os.linesep))
        else:
            for typ in args:
                if node.is_scalar(typ):
                    return
            raise RecognitionError(
                ('{}{}A scalar of type {} is required').format(
                    self.yaml_node.start_mark, os.linesep, args))