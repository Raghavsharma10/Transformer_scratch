def execute(self, conn, child_block_name='', child_lfn_list=[], transaction=False):
        sql = ''
        binds = {}
        child_ds_name = ''
        child_where = ''
        if child_block_name:
            child_ds_name = child_block_name.split('#')[0]
            parent_where = " where d.dataset = :child_ds_name ))"
            binds ={"child_ds_name": child_ds_name}
        else:
            dbsExceptionHandler('dbsException-invalid-input', "Missing child block_name for listFileParentsByLumi. ")
        #
        if not child_lfn_list:
            # most use cases 
            child_where = " where b.block_name = :child_block_name )"
            binds.update({"child_block_name": child_block_name})
            sql = """
            with
            parents as
            (            
            """  +\
            self.parent_sql +\
            parent_where +\
            """), 
 
            """+\
            """
            children as
            (
            """ +\
            self.child_sql +\
            child_where  +\
            """)
            select distinct cid, pid from children c
                inner join parents p on c.R = p.R and c.L = p.L 
            """  
        else:
            # not commom 
            child_where = """ where b.block_name = :child_block_name 
                              and f.logical_file_name in (SELECT TOKEN FROM TOKEN_GENERATOR) ))
                          """
            lfn_generator, bind = create_token_generator(child_lfn_list)
            binds.update(bind)
            sql = lfn_generator +\
            """
            with
            parents as
            (            
            """  +\
            self.parent_sql +\
            parent_where +\
            """), 
 
            """+\
            """
            children as
            (
            """ +\
            self.child_sql +\
            child_where  +\
            """)
            select distinct cid, pid from children c
                inner join parents p on c.R = p.R and c.L = p.L 
            """
        print(sql)


        r = self.dbi.processData(sql, binds, conn, transaction=transaction)
        #print(self.format(r))
        return self.format(r)
        """
        cursors = self.dbi.processData(sql, binds, conn, transaction=transaction, returnCursor=True)
        for i in cursors:
            d = self.formatCursor(i, size=100)
            if isinstance(d, list) or isinstance(d, GeneratorType):
                for elem in d:
                    yield elem
            elif d: 
                yield d
        """