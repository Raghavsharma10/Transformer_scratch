def listBlockSummaries(self, block_name="", dataset="", detail=False):
        """
        API that returns summary information like total size and total number of events in a dataset or a list of blocks

        :param block_name: list block summaries for block_name(s)
        :type block_name: str, list
        :param dataset: list block summaries for all blocks in dataset
        :type dataset: str
        :param detail: list summary by block names if detail=True, default=False
        :type detail: str, bool
        :returns: list of dicts containing total block_sizes, file_counts and event_counts of dataset or blocks provided

        """
        if bool(dataset)+bool(block_name)!=1:
            dbsExceptionHandler("dbsException-invalid-input2",
                                dbsExceptionCode["dbsException-invalid-input2"],
                                self.logger.exception,
                                "Dataset or block_names must be specified at a time.")

        if block_name and isinstance(block_name, basestring):
            try:
                block_name = [str(block_name)]
            except:
                dbsExceptionHandler("dbsException-invalid-input", "Invalid block_name for listBlockSummaries. ")

        for this_block_name in block_name:
            if re.search("[*, %]", this_block_name):
                dbsExceptionHandler("dbsException-invalid-input2",
                                    dbsExceptionCode["dbsException-invalid-input2"],
                                    self.logger.exception,
                                    "No wildcards are allowed in block_name list")

        if re.search("[*, %]", dataset):
            dbsExceptionHandler("dbsException-invalid-input2",
                                dbsExceptionCode["dbsException-invalid-input2"],
                                self.logger.exception,
                                "No wildcards are allowed in dataset")
        data = [] 
        try:
            with self.dbi.connection() as conn:
                data = self.dbsBlockSummaryListDAO.execute(conn, block_name, dataset, detail)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listBlockSummaries. %s\n. Exception trace: \n %s" % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error',
                                dbsExceptionCode['dbsException-server-error'],
                                self.logger.exception,
                                sError)
        for item in data:
                    yield item