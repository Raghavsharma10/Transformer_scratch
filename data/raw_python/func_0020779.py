def listRuns(self, run_num=-1, logical_file_name="", block_name="", dataset=""):
        """
        API to list all runs in DBS. At least one parameter is mandatory.

        :param logical_file_name: List all runs in the file
        :type logical_file_name: str
        :param block_name: List all runs in the block
        :type block_name: str
        :param dataset: List all runs in that dataset
        :type dataset: str
        :param run_num: List all runs
        :type run_num: int, string or list

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
                        dbsExceptionHandler("dbsException-invalid-input", "DBS run range must be apart at least by 1.",
                          self.logger.exception)
                    elif r[0] <= 1 <= r[1]:
                        dbsExceptionHandler("dbsException-invalid-input", "Run_num=1 is not a valid input.",
                                self.logger.exception)
        if run_num==-1 and not logical_file_name and not dataset and not block_name:
                dbsExceptionHandler("dbsException-invalid-input",
                                    "run_num, logical_file_name, block_name or dataset parameter is mandatory",
                                    self.logger.exception)
        try:
            if logical_file_name:
                logical_file_name = logical_file_name.replace("*", "%")
            if block_name:
                block_name = block_name.replace("*", "%")
            if dataset:
                dataset = dataset.replace("*", "%")
            return self.dbsRun.listRuns(run_num, logical_file_name, block_name, dataset)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listRun. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)