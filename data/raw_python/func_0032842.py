def _finalize_database(self, version):
        """
        Create metadata table for database with version number.

        Parameters
        ----------
        version : int
            Tag created database with user-specified version number
        """
        require_integer(version, "version")
        create_metadata_sql = \
            "CREATE TABLE %s (version INT)" % METADATA_TABLE_NAME
        self.execute_sql(create_metadata_sql)
        insert_version_sql = \
            "INSERT INTO %s VALUES (%s)" % (METADATA_TABLE_NAME, version)
        self.execute_sql(insert_version_sql)