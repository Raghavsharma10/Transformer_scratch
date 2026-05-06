def listDataTiers(self, data_tier_name=""):
        """
        List data tier(s)
        """
        if not isinstance(data_tier_name, basestring) :
            dbsExceptionHandler('dbsException-invalid-input',
                                'data_tier_name given is not valid : %s' % data_tier_name)
        else:
            try:
                data_tier_name = str(data_tier_name)
            except:
                dbsExceptionHandler('dbsException-invalid-input',
                                    'data_tier_name given is not valid : %s' % data_tier_name)
        conn = self.dbi.connection()
        try:
            result = self.dataTier.execute(conn, data_tier_name.upper())
            return result
        finally:
            if conn:
                conn.close()