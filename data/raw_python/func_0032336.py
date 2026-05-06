def iter_grants(self, as_json=True):
        """Fetch records from the SQLite database."""
        self._connect()
        result = self.db_connection.cursor().execute(
            "SELECT data, format FROM grants"
        )
        for data, data_format in result:
            if (not as_json) and data_format == 'json':
                raise Exception("Cannot convert JSON source to XML output.")
            elif as_json and data_format == 'xml':
                data = self.grantxml2json(data)
            elif as_json and data_format == 'json':
                data = json.loads(data)
            yield data
        self._disconnect()