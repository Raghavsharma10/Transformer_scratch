def create_database(self, name):
        """Create a new database."""
        statement = "CREATE DATABASE {0} DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci".format(wrap(name))
        return self.execute(statement)