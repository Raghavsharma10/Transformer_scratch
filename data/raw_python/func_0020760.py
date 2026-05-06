def listPrimaryDSTypes(self, primary_ds_type="", dataset=""):
        """
        Returns all primary dataset types if dataset or primary_ds_type are not passed.
        """
        conn = self.dbi.connection()
        try:
            result = self.primdstypeList.execute(conn, primary_ds_type, dataset)
            if conn: conn.close()
            return result
        finally:
            if conn:
                conn.close()