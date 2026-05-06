def listFileChildren(self, logical_file_name='', block_name='', block_id=0):
        """
        required parameter: logical_file_name or block_name or block_id
        returns: logical_file_name, child_logical_file_name, parent_file_id
        """
        conn = self.dbi.connection()
        try:
            if not logical_file_name and not block_name and not block_id:
                dbsExceptionHandler('dbsException-invalid-input',\
                        "Logical_file_name, block_id or block_name is required for listFileChildren api")
            sqlresult = self.filechildlist.execute(conn, logical_file_name, block_name, block_id)
            d = {}
            result = []
            for i in range(len(sqlresult)):
                k = sqlresult[i]['logical_file_name']
                v = sqlresult[i]['child_logical_file_name']
                if k in d:
                    d[k].append(v)
                else:
                    d[k] = [v]
            for k, v in d.iteritems():
                r = {'logical_file_name':k, 'child_logical_file_name': v}
                result.append(r)
            return result
        finally:
            if conn:
                conn.close()