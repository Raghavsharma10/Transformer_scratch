def compose(self, parser: Any, grammar: Any = None, attr_of: str = None):
        """
        Return the CSV(time) expression as string format

        :param parser: Parser instance
        :param grammar: Grammar
        :param attr_of: Attribute of...
        """
        return "CSV({0})".format(self.time)