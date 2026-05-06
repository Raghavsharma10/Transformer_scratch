def update(self,table, sys_id, **kparams):
        """
        use PUT to update a single record by table name and sys_id
        returns a dict (the json map) for python 3.4
        """
        result = self.table_api_put(table, sys_id, **kparams)
        return self.to_record(result, table)