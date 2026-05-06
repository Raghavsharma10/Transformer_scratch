def _get_catalog_url(self, url):
        '''
        Returns the appropriate catalog URL by replacing html with xml in some
        cases
        :param str url: URL to the catalog
        '''
        u = urlparse.urlsplit(url)
        name, ext = os.path.splitext(u.path)
        if ext == ".html":
            u = urlparse.urlsplit(url.replace(".html", ".xml"))
        url = u.geturl()
        return url