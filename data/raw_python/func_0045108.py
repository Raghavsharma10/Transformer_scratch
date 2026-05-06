def list(self,table, **kparams):
        """
        get a collection of records by table name.
        returns a collection of SnowRecord obj.
        """
        records = self.api.list(table, **kparams)
        return records