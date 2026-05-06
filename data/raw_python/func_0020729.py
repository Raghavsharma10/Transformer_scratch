def listRuns(self, run_num=-1, logical_file_name="",
                 block_name="", dataset=""):
        """
        List run known to DBS.
        """
        if( '%' in logical_file_name or '%' in block_name or '%' in dataset ):
            dbsExceptionHandler('dbsException-invalid-input', 
                                " DBSDatasetRun/listRuns. No wildcards are allowed in logical_file_name, block_name or dataset.\n.")
        conn = self.dbi.connection()
        tran = False
        try:
            ret = self.runlist.execute(conn, run_num, logical_file_name, block_name, dataset, tran)
            result = []
            rnum = []
            for i in ret:
                rnum.append(i['run_num'])
            result.append({'run_num' : rnum})
            return result

        finally:
            if conn:
                conn.close()