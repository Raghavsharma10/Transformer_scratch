def backup_db(self):
        """
        " Generate a xxxxx.backup.json.
        """
        with self.db_mutex:
            if os.path.exists(self.json_db_path):
                try:
                    shutil.copy2(self.json_db_path, self.backup_json_db_path)
                except (IOError, OSError):
                    _logger.debug("*** No file to copy.")