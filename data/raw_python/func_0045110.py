def update(self,table, sys_id, **kparams):
        """
        update a record via table api, kparams being the dict of PUT params to update.
        returns a SnowRecord obj.
        """
        record = self.api.update(table, sys_id, **kparams)
        return record