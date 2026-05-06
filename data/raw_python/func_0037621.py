def load_db(self):
        """
        " Load json db as a dictionary.
        """
        try:
            with open(self.json_db_path) as fp:
                self.db = json.load(fp)
        except Exception as e:
            _logger.debug("*** Open JSON DB error.")
            raise e