def analyze(self, rows):
        """
        Analyze the given rows and try to determine the type of value stored.
        :param list rows: A list-of-lists containing one or more rows from a
                          csv file.
        :returns: A list of peewee Field objects for each column in the CSV.
        """
        transposed = zip(*rows)
        checks = self.get_checks()
        column_types = []
        for i, column in enumerate(transposed):
            # Remove any empty values.
            col_vals = [val for val in column if val != '']
            for check in checks:
                results = set(check(val) for val in col_vals)
                if all(results):
                    column_types.append(check.field())
                    break

        return column_types