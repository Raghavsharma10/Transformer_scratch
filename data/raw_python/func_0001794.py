def get_pages(self, url, page=1, page_size=100, yield_pages=False, **filters):
        """
        Get all pages at url, yielding individual results
        :param url: the url to fetch
        :param page: start from this page
        :param page_size: results per page
        :param yield_pages: yield whole pages rather than individual results
        :param filters: additional filters
        :return: a generator of objects (dicts) from the API
        """
        n = 0
        for page in itertools.count(page):
            r = self.request(url, page=page, page_size=page_size, **filters)
            n += len(r['results'])
            log.debug("Got {url} page {page} / {pages}".format(url=url, **r))
            if yield_pages:
                yield r
            else:
                for row in r['results']:
                    yield row
            if r['next'] is None:
                break