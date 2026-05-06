def move_where_clause_to_column(self, column='condition', key=None):
        """
        Move whole WHERE clause to a column named `column`.
        """
        if self.conditions:
            expr = " AND ".join(self.conditions)
            params = self.params
            self.params = []
            self.conditions = []
        else:
            expr = '1'
            params = []
        self.add_column('({0}) AS {1}'.format(expr, column),
                        key or column,
                        params)