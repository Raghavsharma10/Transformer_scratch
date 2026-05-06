def _create_path(self):
        """Create the path to hold the database, if one wwas specified."""

        if self.driver == 'sqlite' and 'memory' not in self.dsn and self.dsn != 'sqlite://':

            dir_ = os.path.dirname(self.path)

            if dir_ and not os.path.exists(dir_):
                try:
                    # Multiple process may try to make, so it could already
                    # exist
                    os.makedirs(dir_)
                except Exception:
                    pass

                if not os.path.exists(dir_):
                    raise Exception("Couldn't create directory " + dir_)