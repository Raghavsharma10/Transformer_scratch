def list(self,table, **kparams):
        """
        get a collection of records by table name.
        returns a dict (the json map) for python 3.4
        """
        result = self.table_api_get(table, **kparams)
        return self.to_records(result, table)