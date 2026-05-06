def show_schema(self, tables=None):
        """Print schema information."""
        tables = tables if tables else self.tables
        for t in tables:
            self._printer('\t{0}'.format(t))
            for col in self.get_schema(t, True):
                self._printer('\t\t{0:30} {1:15} {2:10} {3:10} {4:10} {5:10}'.format(*col))