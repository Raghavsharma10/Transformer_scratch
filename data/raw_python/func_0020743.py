def listBlockParents(self, block_name=""):
        """
        list parents of a block
        """
        if not block_name:
            msg = " DBSBlock/listBlockParents. Block_name must be provided as a string or a list. \
                No wildcards allowed in block_name/s."
            dbsExceptionHandler('dbsException-invalid-input', msg)
        elif isinstance(block_name, basestring):
            try:
                block_name = str(block_name)
                if '%' in block_name or '*' in block_name:
                    dbsExceptionHandler("dbsException-invalid-input", "DBSReaderModel/listBlocksParents: \
                    NO WILDCARDS allowed in block_name.")
            except:
                dbsExceptionHandler("dbsException-invalid-input", "DBSBlock/listBlockParents. Block_name must be \
                provided as a string or a list. No wildcards allowed in block_name/s .")
        elif type(block_name) is list:
            for b in block_name:
                if '%' in b or '*' in b:
                    dbsExceptionHandler("dbsException-invalid-input", "DBSReaderModel/listBlocksParents: \
                            NO WILDCARDS allowed in block_name.")
        else:
            msg = "DBSBlock/listBlockParents. Block_name must be provided as a string or a list. \
                No wildcards allowed in block_name/s ."
            dbsExceptionHandler("dbsException-invalid-input", msg)
        conn = self.dbi.connection()
        try:
            results = self.blockparentlist.execute(conn, block_name)
            return results
        finally:
            if conn:
                conn.close()