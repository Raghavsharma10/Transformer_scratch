def print(self):
        """Print results table.

        >>> Results(['title'], [('Konosuba',), ('Oreimo',)]).print()
          #  title
        ---  --------
          1  Konosuba
          2  Oreimo

        """
        print(tabulate(
            ((i, *row) for i, row in enumerate(self.results, 1)),
            headers=self.headers,
        ))