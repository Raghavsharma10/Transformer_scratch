def page_crawled(self, page_resp):
        """Check if page has been crawled by hashing its text content.

        Add new pages to the page cache.
        Return whether page was found in cache.
        """
        page_text = utils.parse_text(page_resp)
        page_hash = utils.hash_text(''.join(page_text))
        if page_hash not in self.page_cache:
            utils.cache_page(self.page_cache, page_hash, self.args['cache_size'])
            return False
        return True