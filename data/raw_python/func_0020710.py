def listFileSummary(self, block_name="", dataset="", run_num=-1, validFileOnly=0, sumOverLumi=0):
        """
        required parameter: full block_name or dataset name. No wildcards allowed. run_num is optional.
        """
        if not block_name and not dataset:
            msg =  "Block_name or dataset is required for listFileSummary API"
            dbsExceptionHandler('dbsException-invalid-input', msg, self.logger.exception)
        if '%' in block_name or '*' in block_name or '%' in dataset or '*' in dataset:
            msg = "No wildcard is allowed in block_name or dataset for filesummaries API"
            dbsExceptionHandler('dbsException-invalid-input', msg, self.logger.exception)
        #
        with self.dbi.connection() as conn:
            for item in self.filesummarylist.execute(conn, block_name, dataset, run_num,
                validFileOnly=validFileOnly, sumOverLumi=sumOverLumi):
                if item['num_file']==0 and item['num_block']==0 \
                        and item['num_event']==0 and item['file_size']==0:
                    pass
                else:
                    yield item