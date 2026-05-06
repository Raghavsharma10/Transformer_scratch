def listRunSummaries(self, dataset="", run_num=-1):
        """
        API to list run summaries, like the maximal lumisection in a run.

        :param dataset: dataset name (Optional)
        :type dataset: str
        :param run_num: Run number (Required)
        :type run_num: str, long, int
        :rtype: list containing a dictionary with key max_lumi
        """
        if run_num==-1:
            dbsExceptionHandler("dbsException-invalid-input",
                                "The run_num parameter is mandatory",
                                self.logger.exception)

        if re.search('[*,%]', dataset):
            dbsExceptionHandler("dbsException-invalid-input",
                                "No wildcards are allowed in dataset",
                                self.logger.exception)
        # run_num=1 caused full table scan and CERN DBS reported some of the queries ran more than 50 hours
        # We will disbale all the run_num=1 calls in DBS. Run_num=1 will be OK when dataset is given in this API.
        # YG Jan. 16 2019
        if ((run_num == -1 or run_num == '-1') and dataset==''):
            dbsExceptionHandler("dbsException-invalid-input", "Run_num=1 is not a valid input when no dataset is present.", 
                                 self.logger.exception)
        conn = None
        try:
            conn = self.dbi.connection()
            return self.dbsRunSummaryListDAO.execute(conn, dataset, run_num)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listRunSummaries. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'],
                                self.logger.exception, sError)
        finally:
            if conn:
                conn.close()