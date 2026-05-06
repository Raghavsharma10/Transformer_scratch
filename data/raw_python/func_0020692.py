def execute(self, conn, logical_file_name, block_name, block_id, transaction=False):
        """
        Lists all primary datasets if pattern is not provided.
        """
        binds = {}
        sql = ''

        if logical_file_name:
            if isinstance(logical_file_name, basestring):
                wheresql = "WHERE F.LOGICAL_FILE_NAME = :logical_file_name"
                binds = {"logical_file_name": logical_file_name}
                sql = "{sql} {wheresql}".format(sql=self.sql, wheresql=wheresql)
            elif isinstance(logical_file_name, list):
                wheresql = "WHERE F.LOGICAL_FILE_NAME in (SELECT TOKEN FROM TOKEN_GENERATOR)"
                lfn_generator, binds = create_token_generator(logical_file_name)
                sql = "{lfn_generator} {sql} {wheresql}".format(lfn_generator=lfn_generator, sql=self.sql,
                                                                wheresql=wheresql)
        elif block_name:
            joins = "JOIN {owner}BLOCKS B on B.BLOCK_ID = F.BLOCK_ID".format(owner=self.owner)
            wheresql = "WHERE B.BLOCK_NAME = :block_name"
            binds = {"block_name": block_name}
            sql = "{sql} {joins} {wheresql}".format(sql=self.sql, joins=joins, wheresql=wheresql)
        elif block_id:
            wheresql = "WHERE F.BLOCK_ID = :block_id"
            binds = {"block_id": block_id}
            sql = "{sql} {wheresql}".format(sql=self.sql, wheresql=wheresql)
        else:
            dbsExceptionHandler('dbsException-invalid-input', "Logical_file_names is required for listChild dao.", self.logger.exception)

        cursors = self.dbi.processData(sql, binds, conn, transaction=transaction, returnCursor=True)
        result = []
        for c in cursors:
            result.extend(self.formatCursor(c, size=100))
        return result