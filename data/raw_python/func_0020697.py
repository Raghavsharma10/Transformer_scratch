def removeMigrationRequest(self, migration_rqst):
        """
        Method to remove pending or failed migration request from the queue.

        """
        conn = self.dbi.connection()
        try:
            tran = conn.begin()
            self.mgrremove.execute(conn, migration_rqst)
            tran.commit()
        except dbsException as he:
            if conn: conn.close()
            raise
        except Exception as ex:
            if conn: conn.close()
            raise
        if conn: conn.close()