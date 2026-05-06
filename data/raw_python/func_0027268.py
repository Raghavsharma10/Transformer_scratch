def prepare_search_body(self, should_terms=None, must_terms=None, must_not_terms=None, search_text='', start=None, end=None):
        """
        Prepare body for elasticsearch query

        Search parameters
        ^^^^^^^^^^^^^^^^^
        These parameters are dictionaries and have format:  <term>: [<value 1>, <value 2> ...]
        should_terms: it resembles logical OR
        must_terms: it resembles logical AND
        must_not_terms: it resembles logical NOT

        search_text : string
            Text for FTS(full text search)
        start, end : datetime
            Filter for event creation time
        """
        self.body = self.SearchBody()
        self.body.set_should_terms(should_terms)
        self.body.set_must_terms(must_terms)
        self.body.set_must_not_terms(must_not_terms)
        self.body.set_search_text(search_text)
        self.body.set_timestamp_filter(start, end)
        self.body.prepare()