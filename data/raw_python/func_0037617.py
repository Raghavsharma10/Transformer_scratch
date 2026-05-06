def db_manager(self):
        """
        " Do series of DB operations.
        """
        rc_create = self.create_db()    # for first create
        try:
            self.load_db()  # load existing/factory
        except Exception as e:
            _logger.debug("*** %s" % str(e))
            try:
                self.recover_db(self.backup_json_db_path)
            except Exception:
                pass
        else:
            if rc_create is True:
                self.db_status = "factory"
            else:
                self.db_status = "existing"
            return True

        try:
            self.load_db()  # load backup
        except Exception as b:
            _logger.debug("*** %s" % str(b))
            self.recover_db(self.factory_json_db_path)
            self.load_db()  # load factory
            self.db_status = "factory"
        else:
            self.db_status = "backup"
        finally:
            return True