def listAcquisitionEras_CI(self, acq=''):
        """
        Returns all acquistion eras in dbs
        """
        try:
            acq = str(acq)
        except:
            dbsExceptionHandler('dbsException-invalid-input', 'aquistion_era_name given is not valid : %s'%acq)
        conn = self.dbi.connection()
        try:
            result = self.acqlst_ci.execute(conn, acq)
            return result
        finally:
            if conn:conn.close()