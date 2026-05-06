def removeUrl(self, url):
        """Remove passed url from a binder
        """

        root = self.etree
        t_urls = root.find('urls')

        if not t_urls:
            return False

        for t_url in t_urls.findall('url'):
            if t_url.text == url.strip():
                t_urls.remove(t_url)
                if url in self.urls:
                    self.urls.remove(url)
                return True

        return False