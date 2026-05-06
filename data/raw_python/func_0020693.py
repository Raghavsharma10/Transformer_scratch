def listAcquisitionEras(self, acq=''):
        """
        Returns all acquistion eras in dbs
        """
        try:
            acq = str(acq)
        except:
            dbsExceptionHandler('dbsException-invalid-input', 'acquistion_era_name given is not valid : %s' %acq)
        conn = self.dbi.connection()
        try:
            result = self.acqlst.execute(conn, acq)
            return result
        finally:
            if conn:conn.close()