def links(self):
        """Yields all links in the page"""
        for match in self._href_re.finditer(self.content):
            url = match.group(1) or match.group(2) or match.group(3)
            url = self.clean_link(urlparse.urljoin(self.base_url, url))
            yield Link(url, self)