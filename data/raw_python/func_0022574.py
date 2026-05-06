def _parse_query(self, source):
        """Parse one of the rules as either objectfilter or dottysql.

        Example:
            _parse_query("5 + 5")
            # Returns Sum(Literal(5), Literal(5))

        Arguments:
            source: A rule in either objectfilter or dottysql syntax.

        Returns:
            The AST to represent the rule.
        """
        if self.OBJECTFILTER_WORDS.search(source):
            syntax_ = "objectfilter"
        else:
            syntax_ = None  # Default it is.

        return query.Query(source, syntax=syntax_)