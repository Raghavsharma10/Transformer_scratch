def fetch_page_async(self, page_size, **q_options):
    """Fetch a page of results.

    This is the asynchronous version of Query.fetch_page().
    """
    qry = self._fix_namespace()
    return qry._fetch_page_async(page_size, **q_options)