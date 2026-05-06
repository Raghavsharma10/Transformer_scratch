def execute ( self, conn, dataset, dataset_access_type, transaction=False ):
        """
        for a given file
        """
        if not conn:
            dbsExceptionHandler("dbsException-failed-connect2host", "Oracle/Dataset/UpdateType.  Expects db connection from upper layer.", self.logger.exception)
        binds = { "dataset" : dataset , "dataset_access_type" : dataset_access_type ,"myuser": dbsUtils().getCreateBy(), "mydate": dbsUtils().getTime() }
        result = self.dbi.processData(self.sql, binds, conn, transaction)