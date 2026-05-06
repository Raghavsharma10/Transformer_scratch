def update(self, table, items, values):
        """Updates a YQL Table
        >>> yql.update('yql.storage',['value'],['https://josuebrunel.orkg']).where(['name','=','store://YEl70PraLLMSMuYAauqNc7']) 
        """
        self._table = table
        self._limit = None
        items_values = ','.join(["{0} = '{1}'".format(k,v) for k,v in zip(items,values)])
        self._query = "UPDATE {0} SET {1}".format(self._table, items_values)

        return self