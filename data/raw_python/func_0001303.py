def __process_url_wrapper_elements(self, elements):
        """
        Creates the url nodes for pelican.urlwrappers.Category and pelican.urlwrappers.Tag.
        :param elements: list of wrapper elements
        :type elements: list
        :return: the processes urls as HTML
        :rtype: str
        """
        urls = ''
        for url_wrapper, articles in elements:
            urls += self.__create_url_node_for_content(
                url_wrapper,
                'others',
                url=urljoin(self.url_site, url_wrapper.url),
                modification_time=self.__get_date_key(sorted(articles, key=self.__get_date_key, reverse=True)[0])
            )
        return urls