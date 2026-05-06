def extract_html_urls(self, html):
        """
        Take all ``<img src="..">`` from the HTML
        """
        p = HTMLParser(tree=treebuilders.getTreeBuilder("dom"))
        dom = p.parse(html)
        urls = []

        for img in dom.getElementsByTagName('img'):
            src = img.getAttribute('src')
            if src:
                urls.append(unquote_utf8(src))

            srcset = img.getAttribute('srcset')
            if srcset:
                urls += self.extract_srcset(srcset)

        for source in dom.getElementsByTagName('source'):
            srcset = source.getAttribute('srcset')
            if srcset:
                urls += self.extract_srcset(srcset)

        for source in dom.getElementsByTagName('a'):
            href = source.getAttribute('href')
            if href:
                urls.append(unquote_utf8(href))

        return urls