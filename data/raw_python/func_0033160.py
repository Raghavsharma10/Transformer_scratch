def _format(self):
        """Format search queries to perform in bulk.

        Build up the URLs to call for the search engine. These will be ran
        through a bulk processor and returned to a detailer.
        """
        self.log.debug("Formatting URLs to request")
        items = list()
        for i in range(0, self.limit, 10):
            query = '"%s" %s' % (self.domain, self.modifier)
            url = self.host + "/search?q=" + query + "&first=" + str(i)
            items.append(url)
        self.log.debug("URLs were generated")
        return items