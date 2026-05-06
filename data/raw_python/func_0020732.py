def insertBulkBlock(self):
        """
        API to insert a bulk block

        :param blockDump: Output of the block dump command
        :type blockDump: dict

        """
        try:
            body = request.body.read()
            indata = cjson.decode(body)
            if (indata.get("file_parent_list", []) and indata.get("dataset_parent_list", [])): 
                dbsExceptionHandler("dbsException-invalid-input2", "insertBulkBlock: dataset and file parentages cannot be in the input at the same time",  
                    self.logger.exception, "insertBulkBlock: datset and file parentages cannot be in the input at the same time.")    
            indata = validateJSONInputNoCopy("blockBulk", indata)
            self.dbsBlockInsert.putBlock(indata)
        except cjson.DecodeError as dc:
            dbsExceptionHandler("dbsException-invalid-input2", "Wrong format/data from insert BulkBlock input",  self.logger.exception, str(dc))
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.message)
        except HTTPError as he:
            raise he
        except Exception as ex:
            #illegal variable name/number
            if str(ex).find("ORA-01036") != -1:
                dbsExceptionHandler("dbsException-invalid-input2", "illegal variable name/number from input",  self.logger.exception, str(ex))
            else:
                sError = "DBSWriterModel/insertBulkBlock. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
                dbsExceptionHandler('dbsException-server-error',  dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)