def recover_db(self, src_file):
        """
        " Recover DB from xxxxx.backup.json or xxxxx.json.factory to xxxxx.json
        " [src_file]: copy from src_file to xxxxx.json
        """
        with self.db_mutex:
            try:
                shutil.copy2(src_file, self.json_db_path)
            except IOError as e:
                _logger.debug("*** NO: %s file." % src_file)
                raise e