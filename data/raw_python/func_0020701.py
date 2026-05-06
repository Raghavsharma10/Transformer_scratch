def updateMigrationBlockStatus(self, migration_status=0, migration_block=None, migration_request=None):
        """
        migration_status:
        0=PENDING
        1=IN PROGRESS
        2=COMPLETED
        3=FAILED (will be retried)
        9=Terminally FAILED
        status change:
        0 -> 1
        1 -> 2
        1 -> 3
        1 -> 9
        are only allowed changes for working through migration.
        3 -> 1 allowed for retrying.

        """

        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            if migration_block:
                upst = dict(migration_status=migration_status,
                        migration_block_id=migration_block, last_modification_date=dbsUtils().getTime())
            elif migration_request:
                upst = dict(migration_status=migration_status, migration_request_id=migration_request,
                            last_modification_date=dbsUtils().getTime())
            self.mgrup.execute(conn, upst)
        except:
            if tran:tran.rollback()
            raise
        else:
            if tran:tran.commit()
        finally:
            if conn:conn.close()