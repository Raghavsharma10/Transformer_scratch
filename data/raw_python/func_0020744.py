def listBlockChildren(self, block_name=""):
        """
        list parents of a block
        """
        if (not block_name) or re.search("['%','*']", block_name):
            dbsExceptionHandler("dbsException-invalid-input", "DBSBlock/listBlockChildren. Block_name must be provided." )
        conn = self.dbi.connection()
        try:
            results = self.blockchildlist.execute(conn, block_name)
            return results
        finally:
            if conn:
                conn.close()