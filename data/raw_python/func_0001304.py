def __create_url_node_for_content(self, content, content_type, url=None, modification_time=None):
        """
        Creates the required <url> node for the sitemap xml.
        :param content: the content class to handle
        :type content: pelican.contents.Content | None
        :param content_type: the type of the given content to match settings.EXTENDED_SITEMAP_PLUGIN
        :type content_type; str
        :param url; if given, the URL to use instead of the url of the content instance
        :type url: str
        :param modification_time: the modification time of the url, will be used instead of content date if given
        :type modification_time: datetime.datetime | None
        :returns: the text node
        :rtype: str
        """
        loc = url
        if loc is None:
            loc = urljoin(self.url_site, self.context.get('ARTICLE_URL').format(**content.url_format))
        lastmod = None
        if modification_time is not None:
            lastmod = modification_time.strftime('%Y-%m-%d')
        else:
            if content is not None:
                if getattr(content, 'modified', None) is not None:
                    lastmod = getattr(content, 'modified').strftime('%Y-%m-%d')
                elif getattr(content, 'date', None) is not None:
                    lastmod = getattr(content, 'date').strftime('%Y-%m-%d')

        output = "<loc>{}</loc>".format(loc)
        if lastmod is not None:
            output += "\n<lastmod>{}</lastmod>".format(lastmod)
        output += "\n<changefreq>{}</changefreq>".format(self.settings.get('changefrequencies').get(content_type))
        output += "\n<priority>{:.2f}</priority>".format(self.settings.get('priorities').get(content_type))

        return self.template_url.format(output)