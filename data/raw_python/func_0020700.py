def updateMigrationRequestStatus(self, migration_status, migration_request_id):
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
        3 -> 1 is allowed for retrying and retry count +1.

        """

        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            upst = dict(migration_status=migration_status,
                        migration_request_id=migration_request_id,
                        last_modification_date=dbsUtils().getTime())
            self.mgrRqUp.execute(conn, upst)
        except:
            if tran:tran.rollback()
            raise
        else:
            if tran:tran.commit()
        finally:
            #open transaction is committed when conn closed.
            if conn:conn.close()