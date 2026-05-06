def listDatasetParents(self, dataset=""):
        """
        takes required dataset parameter
        returns only parent dataset name
        """
        if( dataset == "" ):
            dbsExceptionHandler("dbsException-invalid-input", "DBSDataset/listDatasetParents. Child Dataset name is required.")
        conn = self.dbi.connection()
        try:
            result = self.datasetparentlist.execute(conn, dataset)
            return result
        finally:
            if conn:
                conn.close()