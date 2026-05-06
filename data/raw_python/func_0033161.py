def _process(self, responses):
        """Process search engine results for detailed analysis.

        Search engine result pages (SERPs) come back with each request and will
        need to be extracted in order to crawl the actual hits.
        """
        self.log.debug("Processing search results")
        items = list()
        for response in responses:
            try:
                soup = BeautifulSoup(response.content, 'html.parser',
                                     from_encoding="iso-8859-1")
            except:
                continue
            else:
                listings = soup.findAll('li', {'class': 'b_algo'})
                items.extend([l.find('a')['href'] for l in listings])
        self.log.debug("Search result URLs were extracted")
        return items