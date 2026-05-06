def insertBlock(self, businput):
        """
        Input dictionary has to have the following keys:
        blockname
        
        It may have:
        open_for_writing, origin_site(name), block_size,
        file_count, creation_date, create_by, last_modification_date, last_modified_by
        
        it builds the correct dictionary for dao input and executes the dao

        NEED to validate there are no extra keys in the businput
        """
        if not ("block_name" in businput and "origin_site_name" in businput  ):
            dbsExceptionHandler('dbsException-invalid-input', "business/DBSBlock/insertBlock must have block_name and origin_site_name as input")
        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            blkinput = {
                "last_modification_date":businput.get("last_modification_date",  dbsUtils().getTime()),
                #"last_modified_by":businput.get("last_modified_by", dbsUtils().getCreateBy()),
                "last_modified_by":dbsUtils().getCreateBy(),
                #"create_by":businput.get("create_by", dbsUtils().getCreateBy()),
                "create_by":dbsUtils().getCreateBy(),
                "creation_date":businput.get("creation_date", dbsUtils().getTime()),
                "open_for_writing":businput.get("open_for_writing", 1),
                "block_size":businput.get("block_size", 0),
                "file_count":businput.get("file_count", 0),
                "block_name":businput.get("block_name"),
                "origin_site_name":businput.get("origin_site_name")
            }
            ds_name = businput["block_name"].split('#')[0]
            blkinput["dataset_id"] = self.datasetid.execute(conn,  ds_name, tran)
            if blkinput["dataset_id"] == -1 : 
                msg = "DBSBlock/insertBlock. Dataset %s does not exists" % ds_name
                dbsExceptionHandler('dbsException-missing-data', msg)
            blkinput["block_id"] =  self.sm.increment(conn, "SEQ_BK", tran)
            self.blockin.execute(conn, blkinput, tran)

            tran.commit()
            tran = None
        except Exception as e:
            if str(e).lower().find("unique constraint") != -1 or str(e).lower().find("duplicate") != -1:
                pass
            else:
                if tran:
                    tran.rollback()
                if conn: conn.close()
                raise
                
        finally:
            if tran:
                tran.rollback()
            if conn:
                conn.close()