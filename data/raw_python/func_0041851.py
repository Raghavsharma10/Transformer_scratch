def checkForeignKeys(self, engine: Engine) -> None:
        """ Check Foreign Keys

        Log any foreign keys that don't have indexes assigned to them.
        This is a performance issue.

        """
        missing = (sqlalchemy_utils.functions
                   .non_indexed_foreign_keys(self._metadata, engine=engine))

        for table, keys in missing.items():
            for key in keys:
                logger.warning("Missing index on ForeignKey %s" % key.columns)