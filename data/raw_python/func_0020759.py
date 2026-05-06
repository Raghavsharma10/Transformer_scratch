def listPrimaryDatasets(self, primary_ds_name="", primary_ds_type=""):
        """
        Returns all primary dataset if primary_ds_name or primary_ds_type are not passed.
        """
        conn = self.dbi.connection()
        try:
            result = self.primdslist.execute(conn, primary_ds_name, primary_ds_type)
            if conn: conn.close()
            return result
        finally:
            if conn:
                conn.close()