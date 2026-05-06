def insert(self, table,items, values):
        """This method allows to insert data into table
        >>> yql.insert('bi.ly.shorten',('login','apiKey','longUrl'),('YOUR LOGIN','YOUR API KEY','YOUR LONG URL'))
        """
        values = ["'{0}'".format(e) for e in values]
        self._query = "INSERT INTO {0} ({1}) VALUES ({2})".format(table,','.join(items),','.join(values))
        payload = self._payload_builder(self._query)
        response = self.execute_query(payload)

        return response