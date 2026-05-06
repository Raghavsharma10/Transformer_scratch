def listProcessingEras(self, processing_version=''):
        """
        Returns all processing eras in dbs
        """
        conn = self.dbi.connection()
        try:
            result = self.pelst.execute(conn, processing_version)
            return result
        finally:
            if conn:
                conn.close()