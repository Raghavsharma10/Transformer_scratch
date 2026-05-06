def _involvedTables(self):
        """
        Return a list of tables involved in this query,
        first checking that no required tables (those in
        the query target) have been omitted from the comparison.
        """
        # SQL and arguments
        if self.comparison is not None:
            tables = self.comparison.getInvolvedTables()
            self.args = self.comparison.getArgs(self.store)
        else:
            tables = list(self.tableClass)
            self.args = []

        for tableClass in self.tableClass:
            if tableClass not in tables:
                raise ValueError(
                    "Comparison omits required reference to result type %s"
                    % tableClass.typeName)

        return tables