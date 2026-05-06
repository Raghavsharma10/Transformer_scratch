def execute(self, conn, site_name= "", transaction = False):
        """
        Lists all sites types if site_name is not provided.
        """
        sql = self.sql
        if site_name == "":
            result = self.dbi.processData(sql, conn=conn, transaction=transaction)
        else:
            sql += "WHERE S.SITE_NAME = :site_name" 
            binds = { "site_name" : site_name }
            result = self.dbi.processData(sql, binds, conn, transaction)
        return self.formatDict(result)