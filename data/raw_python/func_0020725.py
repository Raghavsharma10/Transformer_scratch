def updateStatus(self, dataset, is_dataset_valid):
        """
        Used to toggle the status of a dataset  is_dataset_valid=0/1 (invalid/valid)
        """
        if( dataset == "" ):
            dbsExceptionHandler("dbsException-invalid-input", "DBSDataset/updateStatus. dataset is required.")

        conn = self.dbi.connection()
        trans = conn.begin()

        try:
            self.updatestatus.execute(conn, dataset, is_dataset_valid, trans)
            trans.commit()
            trans = None
        except Exception as ex:
            if trans:
                trans.rollback()
            raise ex
        finally:
            if trans:
                trans.rollback()
            if conn:
                conn.close()