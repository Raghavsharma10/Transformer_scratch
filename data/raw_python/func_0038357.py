def schema(self, table):
        """
        Print the table schema

        Parameters
        ----------
        table: str
          The table name

        """
        try:
            pprint(self.query("PRAGMA table_info({})".format(table), fmt='table'))
        except ValueError:
            print('Table {} not found'.format(table))