def listMigrationRequests(self, migration_request_id="", block_name="",
                              dataset="", user="", oldest=False):
        """
        get the status of the migration
        migratee : can be dataset or block_name
        """

        conn = self.dbi.connection()
        migratee = ""
        try:
            if block_name:
                migratee = block_name
            elif dataset:
                migratee = dataset
            result = self.mgrlist.execute(conn, migration_url="",
                    migration_input=migratee, create_by=user,
                    migration_request_id=migration_request_id, oldest=oldest)
            return result

        finally:
            if conn: conn.close()