def explain(self):
        """
        A debugging API, exposing SQLite's I{EXPLAIN} statement.

        While this is not a private method, you also probably don't have any
        use for it unless you understand U{SQLite
        opcodes<http://www.sqlite.org/opcode.html>} very well.

        Once you do, it can be handy to call this interactively to get a sense
        of the complexity of a query.

        @return: a list, the first element of which is a L{str} (the SQL
        statement which will be run), and the remainder of which is 3-tuples
        resulting from the I{EXPLAIN} of that statement.
        """
        return ([self._sqlAndArgs('SELECT', self._queryTarget)[0]] +
                self._runQuery('EXPLAIN SELECT', self._queryTarget))