def updateType(self, dataset, dataset_access_type):
        """
        Used to change the status of a dataset type (production/etc.)
        """
        if( dataset == "" ):
            dbsExceptionHandler("dbsException-invalid-input", "DBSDataset/updateType. dataset is required.")

        conn = self.dbi.connection()
        trans = conn.begin()

        try :
            self.updatetype.execute(conn, dataset, dataset_access_type.upper(), trans)
            trans.commit()
            trans = None
        except SQLAlchemyDatabaseError as ex:
            if str(ex).find("ORA-01407") != -1:
                dbsExceptionHandler("dbsException-invalid-input2", "Invalid Input", None, "DBSDataset/updateType. A Valid dataset_access_type is required.")
        finally:
            if trans:
                trans.rollback()
            if conn:
                conn.close()