def reload(self):
        """
        Rerun the query (lazily).
        The results will contain any values on the server side that have changed since the last run.
        :return: None
        """
        self._results = []
        self._next_item_index = 0
        self._next_page_index = 0
        self._last_page_seen = False