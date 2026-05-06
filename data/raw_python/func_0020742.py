def updateSiteName(self, block_name, origin_site_name):
        """
        Update the origin_site_name for a given block name
        """
        if not origin_site_name:
            dbsExceptionHandler('dbsException-invalid-input',
                                "DBSBlock/updateSiteName. origin_site_name is mandatory.")
        conn = self.dbi.connection()
        trans = conn.begin()
        try:
            self.updatesitename.execute(conn, block_name, origin_site_name)
        except:
            if trans:
                trans.rollback()
            raise
        else:
            if trans:
                trans.commit()
        finally:
            if conn:
                conn.close()