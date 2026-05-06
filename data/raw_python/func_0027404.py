def _run(self, url, auth):
        '''
        Performs a multiprocess depth-first-search of the catalog references
        and yields a URL for each leaf dataset found
        :param str url: URL for the current catalog
        :param requests.auth.AuthBase auth: requets auth object to use
        '''
        if url in self.visited:
            logger.debug("Skipping %s (already crawled)" % url)
            return
        self.visited.append(url)

        logger.info("Crawling: %s" % url)
        url = self._get_catalog_url(url)

        # Get an etree object
        xml_content = request_xml(url, auth)
        for ds in self._build_catalog(url, xml_content):
            yield ds