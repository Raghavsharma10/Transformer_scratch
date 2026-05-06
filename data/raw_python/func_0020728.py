def execute(self, conn, block_name, origin_site_name, transaction=False):
        """
        Update origin_site_name for a given block_name
        """
        if not conn:
            dbsExceptionHandler("dbsException-failed-connect2host", "Oracle/Block/UpdateStatus. \
Expects db connection from upper layer.", self.logger.exception)
        binds = {"block_name": block_name, "origin_site_name": origin_site_name, "mtime": dbsUtils().getTime(),
                 "myuser": dbsUtils().getCreateBy()}
        self.dbi.processData(self.sql, binds, conn, transaction)