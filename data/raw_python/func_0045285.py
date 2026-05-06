def get(self,table, sys_id):
        """
        get a single record by table name and sys_id
        returns a dict (the json map) for python 3.4
        """
        result = self.table_api_get(table, sys_id)
        return self.to_record(result, table)