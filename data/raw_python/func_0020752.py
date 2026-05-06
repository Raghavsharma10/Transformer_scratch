def getBufferedFiles(self, block_id):
        """
        Get some files from the insert buffer
        """
            
        try:
            conn = self.dbi.connection()
            result = self.buflist.execute(conn, block_id)
            return result
        finally:
            if conn:
                conn.close()