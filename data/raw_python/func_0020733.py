def insertBlock(self):
        """
        API to insert a block into DBS

        :param blockObj: Block object
        :type blockObj: dict
        :key open_for_writing: Open For Writing (1/0) (Optional, default 1)
        :key block_size: Block Size (Optional, default 0)
        :key file_count: File Count (Optional, default 0)
        :key block_name: Block Name (Required)
        :key origin_site_name: Origin Site Name (Required)

        """
        try:
            body = request.body.read()
            indata = cjson.decode(body)
            indata = validateJSONInputNoCopy("block", indata)
            self.dbsBlock.insertBlock(indata)
        except cjson.DecodeError as dc:
            dbsExceptionHandler("dbsException-invalid-input2", "Wrong format/data from insert Block input",  self.logger.exception, str(dc))
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.message)
        except Exception as ex:
            sError = "DBSWriterModel/insertBlock. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error',  dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)