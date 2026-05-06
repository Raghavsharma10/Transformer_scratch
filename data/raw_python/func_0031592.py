def index_all(self):
        """
        Index all records under :attr:`record_path`.
        """
        self.logger.debug('Start indexing all records under: %s',
                          self.record_path)
        with self.db.connection():
            for json_path in sorted(self.find_record_files()):
                self.index_record(json_path)