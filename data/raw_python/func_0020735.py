def updateFile(self, logical_file_name=[], is_file_valid=1, lost=0, dataset=''):
        """
        API to update file status

        :param logical_file_name: logical_file_name to update (optional), but must have either a fln or 
        a dataset
        :type logical_file_name: str
        :param is_file_valid: valid=1, invalid=0 (Required)
        :type is_file_valid: bool
        :param lost: default lost=0 (optional)
        :type lost: bool
        :param dataset: default dataset='' (optional),but must have either a fln or a dataset
        :type dataset: basestring

        """
        if lost in [1, True, 'True', 'true', '1', 'y', 'yes']:
            lost = 1
            if is_file_valid in [1, True, 'True', 'true', '1', 'y', 'yes']:
                dbsExceptionHandler("dbsException-invalid-input2", dbsExceptionCode["dbsException-invalid-input2"], self.logger.exception,\
                                    "Lost file must set to invalid" )
        else: lost = 0
        
        for f in logical_file_name, dataset:
            if '*' in f or '%' in f:
                dbsExceptionHandler("dbsException-invalid-input2", dbsExceptionCode["dbsException-invalid-input2"], self.logger.exception, "No \
                    wildcard allow in LFN or dataset for updatefile API." )
        try:
            self.dbsFile.updateStatus(logical_file_name, is_file_valid, lost, dataset)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSWriterModel/updateFile. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error',  dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)