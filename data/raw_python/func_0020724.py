def listDatasetChildren(self, dataset):
        """
        takes required dataset parameter
        returns only children dataset name
        """
        if( dataset == "" ):
            dbsExceptionHandler("dbsException-invalid-input", "DBSDataset/listDatasetChildren. Parent Dataset name is required.")
        conn = self.dbi.connection()
        try:
            result = self.datasetchildlist.execute(conn, dataset)
            return result
        finally:
            if conn:
                conn.close()