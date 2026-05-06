def compose(self, parser: Any, grammar: Any = None, attr_of: str = None) -> str:
        """
        Return the Condition as string format

        :param parser: Parser instance
        :param grammar: Grammar
        :param attr_of: Attribute of...
        """
        if type(self.left) is Condition:
            left = "({0})".format(parser.compose(self.left, grammar=grammar, attr_of=attr_of))
        else:
            left = parser.compose(self.left, grammar=grammar, attr_of=attr_of)

        if getattr(self, 'op', None):

            if type(self.right) is Condition:
                right = "({0})".format(parser.compose(self.right, grammar=grammar, attr_of=attr_of))
            else:
                right = parser.compose(self.right, grammar=grammar, attr_of=attr_of)
            op = parser.compose(self.op, grammar=grammar, attr_of=attr_of)
            result = "{0} {1} {2}".format(left, op, right)
        else:
            result = left
        return result