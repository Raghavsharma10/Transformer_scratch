def query(self):
        """Parse query string using given grammar.

        :returns: AST that represents the query in the given grammar.
        """
        tree = pypeg2.parse(self._query, parser(), whitespace="")
        for walker in query_walkers():
            tree = tree.accept(walker)
        return tree