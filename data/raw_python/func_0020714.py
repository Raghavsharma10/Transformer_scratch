def updateStatus(self, logical_file_name, is_file_valid, lost, dataset):
        """
        Used to toggle the status of a file from is_file_valid=1 (valid) to is_file_valid=0 (invalid)
        """

        conn = self.dbi.connection()
        trans = conn.begin()
        try :
            self.updatestatus.execute(conn, logical_file_name, is_file_valid, lost, dataset, trans)
            trans.commit()
            trans = None
        except Exception as ex:
            if trans:
                trans.rollback()
                trans = None
            raise ex

        finally:
            if trans:
                trans.rollback()
            if conn:
                conn.close()