def more_search(self, more_page):
        """Method to add more result to an already exist result.

        more_page determine how many result page should be added
        to the current result.
        """
        next_page = self.current_page + 1
        top_page = more_page + self.current_page
        for page in range(next_page, (top_page + 1)):
            start = "start={0}".format(str((page - 1) * 10))
            url = "{0}{1}&{2}".format(self.google, self.query, start)
            self._execute_search_request(url)
            self.current_page += 1