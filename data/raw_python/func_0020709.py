def listFileLumis(self, logical_file_name="", block_name="", run_num=-1, validFileOnly=0, input_body=-1):
        """
        optional parameter: logical_file_name, block_name, validFileOnly
        returns: logical_file_name, file_lumi_id, run_num, lumi_section_num
        """
        if((logical_file_name=='' or '*'in logical_file_name or '%' in logical_file_name) \
            and (block_name=='' or '*' in block_name or '%' in block_name) and input_body==-1 ):
            dbsExceptionHandler('dbsException-invalid-input', \
                "Fully specified logical_file_name or block_name is required if GET is called. No wildcards are allowed.",
		 self.logger.exception, "Fully specified logical_file_name or block_name is required if GET is called. No wildcards are allowed.")
        elif input_body != -1 :
            try:
                logical_file_name = input_body["logical_file_name"]
                run_num = input_body.get("run_num", -1)
                validFileOnly = input_body.get("validFileOnly", 0)
                block_name = ""
            except cjson.DecodeError as de:
                msg = "business/listFileLumis requires at least a list of logical_file_name. %s" % de
                dbsExceptionHandler('dbsException-invalid-input2', "Invalid input", self.logger.exception, msg)
        elif input_body != -1 and (logical_file_name is not None or block_name is not None): 
            dbsExceptionHandler('dbsException-invalid-input', "listFileLumis may have input in the command or in the payload, not mixed.", self.logger.exception, "listFileLumis may have input in the command or in the payload, not mixed.")

        with self.dbi.connection() as conn:
            for item in  self.filelumilist.execute(conn, logical_file_name, block_name, run_num, validFileOnly=validFileOnly):
                yield item