def getBlocks(self):
        """
        Get the blocks that need to be migrated
        """
        try:
            conn = self.dbi.connection()
            result = self.buflistblks.execute(conn)
            return result
        finally:
            if conn:
                conn.close()