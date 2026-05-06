def execute(self, conn, logical_file_name='', block_id=0, block_name='', transaction=False):
        """
        return {} if condition is not provided.
        """
        sql = ''
        binds = {}

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
        elif block_id != 0:
            wheresql = "WHERE F.BLOCK_ID = :block_id"
            binds ={'block_id': block_id}
            sql = "{sql} {wheresql}".format(sql=self.sql, wheresql=wheresql)
        elif block_name:
            joins = "JOIN {owner}BLOCKS B on B.BLOCK_ID = F.BLOCK_ID".format(owner=self.owner)
            wheresql = "WHERE B.BLOCK_NAME= :block_name"
            binds ={'block_name': block_name}
            sql = "{sql} {joins} {wheresql}".format(sql=self.sql, joins=joins, wheresql=wheresql)
        else:
            return

        cursors = self.dbi.processData(sql, binds, conn, transaction=transaction, returnCursor=True)
        for i in cursors:
            d = self.formatCursor(i, size=100)
            if isinstance(d, list) or isinstance(d, GeneratorType):
                for elem in d:
                    yield elem
            elif d: 
                yield d