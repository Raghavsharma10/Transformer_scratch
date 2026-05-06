def get(self,table, sys_id):
        """
        get a single record by table name and sys_id
        returns a SnowRecord obj.
        """
        record = self.api.get(table, sys_id)
        return record