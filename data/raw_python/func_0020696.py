def UpdateAcqEraEndDate(self, acquisition_era_name ="", end_date=0):
        """
        Input dictionary has to have the following keys:
        acquisition_era_name, end_date.
        """
        if acquisition_era_name =="" or end_date==0:
            dbsExceptionHandler('dbsException-invalid-input', "acquisition_era_name and end_date are required")
        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            self.acqud.execute(conn, acquisition_era_name, end_date, tran)
            if tran:tran.commit()
            tran = None
        finally:
            if tran:tran.rollback()
            if conn:conn.close()