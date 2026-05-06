def save_db(self):
        """
        " Save json db to file system.
        """
        with self.db_mutex:
            if not isinstance(self.db, dict) and not isinstance(self.db, list):
                return False
            try:
                with open(self.json_db_path, "w") as fp:
                    json.dump(self.db, fp, indent=4)
            except Exception as e:
                # disk full or something.
                _logger.debug("*** Write JSON DB to file error.")
                raise e

            else:
                self.sync()
                return True