def generate_output(self, writer):
        """
        Generates the sitemap file and the stylesheet file and puts them into the content dir.
        :param writer: the writer instance
        :type writer: pelican.writers.Writer
        """
        # write xml stylesheet
        with codecs_open(os.path.join(os.path.dirname(__file__), 'sitemap-stylesheet.xsl'), 'r', encoding='utf-8') as fd_origin:
            with codecs_open(os.path.join(self.path_output, 'sitemap-stylesheet.xsl'), 'w', encoding='utf-8') as fd_destination:
                xsl = fd_origin.read()
                # replace some template markers
                # TODO use pelican template magic
                xsl = xsl.replace('{{ SITENAME }}', self.context.get('SITENAME'))
                fd_destination.write(xsl)

        # will contain the url nodes as text
        urls = ''

        # get all articles sorted by time
        articles_sorted = sorted(self.context['articles'], key=self.__get_date_key, reverse=True)

        # get all pages with date/modified date
        pages_with_date = list(
            filter(
                lambda p: getattr(p, 'modified', False) or getattr(p, 'date', False),
                self.context.get('pages')
            )
        )
        pages_with_date_sorted = sorted(pages_with_date, key=self.__get_date_key, reverse=True)

        # get all pages without date
        pages_without_date = list(
            filter(
                lambda p: getattr(p, 'modified', None) is None and getattr(p, 'date', None) is None,
                self.context.get('pages')
            )
        )
        pages_without_date_sorted = sorted(pages_without_date, key=self.__get_title_key, reverse=False)

        # join them, first date sorted, then title sorted
        pages_sorted = pages_with_date_sorted + pages_without_date_sorted

        # the landing page
        if 'index' in self.context.get('DIRECT_TEMPLATES'):
            # assume that the index page has changed with the most current article or page
            # use the first article or page if no articles
            index_reference = None
            if len(articles_sorted) > 0:
                index_reference = articles_sorted[0]
            elif len(pages_sorted) > 0:
                index_reference = pages_sorted[0]

            if index_reference is not None:
                urls += self.__create_url_node_for_content(
                    index_reference,
                    'index',
                    url=self.url_site,
                )

        # process articles
        for article in articles_sorted:
            urls += self.__create_url_node_for_content(
                article,
                'articles',
                url=urljoin(self.url_site, article.url)
            )

        # process pages
        for page in pages_sorted:
            urls += self.__create_url_node_for_content(
                page,
                'pages',
                url=urljoin(self.url_site, page.url)
            )

        # process category pages
        if self.context.get('CATEGORY_URL'):
            urls += self.__process_url_wrapper_elements(self.context.get('categories'))

        # process tag pages
        if self.context.get('TAG_URL'):
            urls += self.__process_url_wrapper_elements(sorted(self.context.get('tags'), key=lambda x: x[0].name))

        # process author pages
        if self.context.get('AUTHOR_URL'):
            urls += self.__process_url_wrapper_elements(self.context.get('authors'))

        # handle all DIRECT_TEMPLATES but "index"
        for direct_template in list(filter(lambda p: p != 'index', self.context.get('DIRECT_TEMPLATES'))):
            # we assume the modification date of the last article as modification date for the listings of
            # categories, authors and archives (all values of DIRECT_TEMPLATES but "index")
            modification_time = getattr(articles_sorted[0], 'modified', getattr(articles_sorted[0], 'date', None))
            url = self.__get_direct_template_url(direct_template)
            urls += self.__create_url_node_for_content(None, 'others', url, modification_time)

        # write the final sitemap file
        with codecs_open(os.path.join(self.path_output, 'sitemap.xml'), 'w', encoding='utf-8') as fd:
            fd.write(self.xml_wrap % {
                'SITEURL': self.url_site,
                'urls': urls
            })