def fromHTML(html, *args, **kwargs):
        """
        Creates abstraction using HTML

        :param str html: HTML
        :return: TreeOfContents object
        """
        source = BeautifulSoup(html, 'html.parser', *args, **kwargs)
        return TOC('[document]',
            source=source,
            descendants=source.children)