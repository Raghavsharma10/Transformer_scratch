def get_scroll(self, url, page_size=100, yield_pages=False, **filters):
        """
        Scroll through the resource at url and yield the individual results
        :param url: url to scroll through
        :param page_size: results per page
        :param yield_pages: yield whole pages rather than individual results
        :param filters: Additional filters
        :return: a generator of objects (dicts) from the API
        """
        n = 0
        options = dict(page_size=page_size, **filters)
        format = filters.get('format')
        while True:
            r = self.request(url, use_xpost=False, **options)
            n += len(r['results'])
            log.debug("Got {} {n}/{total}".format(url.split("?")[0], total=r['total'], **locals()))
            if yield_pages:
                yield r
            else:
                for row in r['results']:
                    yield row
            if r['next'] is None:
                break
            url = r['next']
            options = {'format': None}