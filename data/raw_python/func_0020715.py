def insertFileParents(self, businput):
        """
        This is a special function for WMAgent only.
        input block_name: is a child block name.
        input chils_parent_id_list: is a list of file id of child, parent  pair: [[cid1, pid1],[cid2,pid2],[cid3,pid3],...]        
        The requirment for this API is 
        1. All the child files belong to the block.
        2. All the child-parent pairs are not already in DBS.
        3. The dataset parentage is already in DBS.
        We will fill the block parentage here using the file parentage info.
      
        Y. Guo 
        July 18, 2018 
        """
        if "block_name" not in businput.keys() or "child_parent_id_list" not in businput.keys() or not businput["child_parent_id_list"] or not businput["block_name"]:
            dbsExceptionHandler("dbsException-invalid-input2", "DBSFile/insertFileParents: require child block_name and list of child/parent file id pairs" , self.logger.exception, "DBSFile/insertFileParents: require child block_name and list of child/parent file id pairs")
        tran = None
        conn = None  
        try:
            #We should get clean insert for both file/block parentage.
            #block parent duplication is handled at dao level. File parent should not have deplication.  
            conn = self.dbi.connection()
            tran = conn.begin()
            self.logger.info("Insert File parentage mapping") 
            self.fparentin2.execute(conn, businput, tran)
            self.logger.info("Insert block parentage mapping")
            self.blkparentin3.execute(conn, businput, tran)
            if tran:tran.commit()
            if conn:conn.close()
        except SQLAlchemyIntegrityError as ex:
                if tran:tran.rollback()
                if conn:conn.close()
                if str(ex).find("ORA-01400") > -1:
                    dbsExceptionHandler('dbsException-missing-data',
                        'Missing data when insert filei/block parent. ', self.logger.exception,
                        'Missing data when insert file/block parent. '+ str(ex))
                else:
                    dbsExceptionHandler('dbsException-invalid-input2',
                        'Invalid data when insert file/block parent.  ', self.logger.exception,
                        'Invalid data when insert file/block parent. '+ str(ex))
        finally:
            if tran:tran.rollback()
            if conn:conn.close()