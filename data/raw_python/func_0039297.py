def addUrl(self, url):
        """Add url to binder
        """

        if url not in self.urls:
            self.urls.append(url)

        root = self.etree
        t_urls = root.find('urls')

        if not t_urls:
            t_urls = ctree.SubElement(root, 'urls')

        t_url = ctree.SubElement(t_urls, 'url')
        t_url.text = url

        return True