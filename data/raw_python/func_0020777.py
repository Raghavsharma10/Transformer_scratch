def listFileChildren(self, logical_file_name='', block_name='', block_id=0):
        """
        API to list file children. One of the parameters in mandatory.

        :param logical_file_name: logical_file_name of file (Required)
        :type logical_file_name: str, list
        :param block_name: block_name
        :type block_name: str
        :param block_id: block_id
        :type block_id: str, int
        :returns: List of dictionaries containing the following keys (child_logical_file_name, logical_file_name)
        :rtype: List of dicts

        """
        if isinstance(logical_file_name, list):
            for f in logical_file_name:
                if '*' in f or '%' in f:
                    dbsExceptionHandler("dbsException-invalid-input2", dbsExceptionCode["dbsException-invalid-input2"], self.logger.exception, "No \
                                         wildcard allow in LFN list" )

        try:
            return self.dbsFile.listFileChildren(logical_file_name, block_name, block_id)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listFileChildren. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)