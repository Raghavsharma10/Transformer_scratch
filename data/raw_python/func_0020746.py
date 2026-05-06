def listBlocksOrigin(self, origin_site_name="", dataset="", block_name=""):
        """
        This is the API to list all the blocks/datasets first generated in the site called origin_site_name,
        if origin_site_name is provided w/ no wildcards allow. If a fully spelled dataset is provided, then it will
        only list the blocks first generated from origin_site_name under the given dataset.
        """
        if not (dataset or block_name):
            dbsExceptionHandler("dbsException-invalid-input",
                                "DBSBlock/listBlocksOrigin: dataset or block_name must be provided.")
        if re.search("['%', '*']", dataset) or re.search("['%', '*']", block_name):
            dbsExceptionHandler("dbsException-invalid-input",
                                "DBSBlock/listBlocksOrigin: dataset or block_name with wildcard is not supported.")
        try:
            conn = self.dbi.connection()
            result = self.bkOriginlist.execute(conn, origin_site_name, dataset, block_name)
            return result
        finally:
            if conn:
                conn.close()