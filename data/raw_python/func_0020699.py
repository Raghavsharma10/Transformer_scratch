def listMigrationBlocks(self, migration_request_id=""):
        """
        get eveything of block that is has status = 0 and migration_request_id as specified.
        """

        conn = self.dbi.connection()
        try:
            return self.mgrblklist.execute(conn, migration_request_id=migration_request_id)
        finally:
            if conn: conn.close()