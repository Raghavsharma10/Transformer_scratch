def _fetch_result(self):
        """
        Fetch the queried object.
        """
        self._result = self.conn.query_single(self.object_type, self.url_params, self.query_params)