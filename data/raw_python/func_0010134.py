def get_new_links(self, url, resp):
        """Get new links from a URL and filter them."""
        links_on_page = resp.xpath('//a/@href')
        links = [utils.clean_url(u, url) for u in links_on_page]

        # Remove non-links through filtering by protocol
        links = [x for x in links if utils.check_protocol(x)]

        # Restrict new URLs by the domain of the input URL
        if not self.args['nonstrict']:
            domain = utils.get_domain(url)
            links = [x for x in links if utils.get_domain(x) == domain]

        # Filter URLs by regex keywords, if any
        if self.args['crawl']:
            links = utils.re_filter(links, self.args['crawl'])
        return links