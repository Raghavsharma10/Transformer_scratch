def listDataType(self, dataType="", dataset=""):
        """
        List data-type/primary-ds-type 
        """
        conn = self.dbi.connection()
        try:
            if dataset and dataType:
                dbsExceptionHandler('dbsException-invalid-input',
                    "DBSDataType/listDataType. Data Type can be only searched by data_type or by dataset, not both.")
            else:
                result = self.dataType.execute(conn, dataType, dataset)
                return result
        finally:
            if conn:
                conn.close()