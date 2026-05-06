def listSites(self, block_name="", site_name=""):
        """
        Returns sites.
        """
        try:
            conn = self.dbi.connection()
            if block_name:
                result = self.blksitelist.execute(conn, block_name)
            else:
                result = self.sitelist.execute(conn, site_name)
            return result
        finally:
            if conn:
                conn.close()