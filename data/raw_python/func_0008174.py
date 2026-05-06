def _next_page(self):
        """
        Fetch the next page of the query.
        """
        if self._last_page_seen:
            raise StopIteration
        new, self._last_page_seen = self.conn.query_multiple(self.object_type, self._next_page_index,
                                                             self.url_params, self.query_params)
        self._next_page_index += 1
        if len(new) == 0:
            self._last_page_seen = True  # don't bother with next page if nothing was returned
        else:
            self._results += new