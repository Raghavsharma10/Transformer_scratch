def start_search(self, max_page=1):
        """method to start send query to google. Search start from page 1.

        max_page determine how many result expected
        hint: 10 result per page for google
        """
        for page in range(1, (max_page + 1)):
            start = "start={0}".format(str((page - 1) * 10))
            url = "{0}{1}&{2}".format(self.google, self.query, start)
            self._execute_search_request(url)
            self.current_page += 1