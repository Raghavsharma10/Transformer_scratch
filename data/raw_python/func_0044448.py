def _clean_url(self, url):
        """
        Canonicalizes the url, as it is done in Scrapy.
        And keeps only USEFUL_QUERY_KEYS. It also strips the 
        trailing slash to help identifying dupes.
        """
        # TODO: Turn this into regex
        if not url.startswith('http') or url.endswith('}}') or 'nojs_router' in url:
            return None
        if site(norm(url).lower()) in config.NONCANONIC_SITES:
            clean_url = canonicalize_url(url, keep_params=True)
        else:
            clean_url = canonicalize_url(url)
        return clean_url