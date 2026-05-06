def listFileLumis(self, logical_file_name="", block_name="", run_num=-1, validFileOnly=0):
        """
        API to list Lumi for files. Either logical_file_name or block_name is required. No wild card support in this API

        :param block_name: Name of the block
        :type block_name: str
        :param logical_file_name: logical_file_name of file
        :type logical_file_name: str, list
        :param run_num: List lumi sections for a given run number (Optional). Possible format are: run_num, 'run_min-run_max' or ['run_min-run_max', run1, run2, ...]. run_num=1 is for MC data and caused almost full table scan. So run_num=1
                        will cause an input error.
        :type run_num: int, str, or list
        :returns: List of dictionaries containing the following keys (lumi_section_num, logical_file_name, run_num, event_count)
        :rtype: list of dicts
        :param validFileOnly: optional valid file flag. Default = 0 (include all files)
        :type: validFileOnly: int, or str

        """
        # run_num=1 caused full table scan and CERN DBS reported some of the queries ran more than 50 hours
        # We will disbale all the run_num=1 calls in DBS. Run_num=1 will be OK when logical_file_name is given.
        # YG Jan. 16 2019
        if (run_num != -1  and logical_file_name ==''):
            for r in parseRunRange(run_num):
                if isinstance(r, basestring) or isinstance(r, int) or isinstance(r, long):    
                    if r == 1 or r == '1':
                        dbsExceptionHandler("dbsException-invalid-input", "Run_num=1 is not a valid input.",
                                self.logger.exception)
                elif isinstance(r, run_tuple):
                    if r[0] == r[1]:
                        dbsExceptionHandler("dbsException-invalid-input", "DBS run range must be apart at least by 1.",self.logger.exception)
                    elif r[0] <= 1 <= r[1]:
                        dbsExceptionHandler("dbsException-invalid-input", "Run_num=1 is not a valid input.",
                                self.logger.exception) 
        try:
            return self.dbsFile.listFileLumis(logical_file_name, block_name, run_num, validFileOnly )
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listFileLumis. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)