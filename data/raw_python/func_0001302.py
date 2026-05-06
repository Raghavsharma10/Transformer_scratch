def __get_direct_template_url(self, name):
        """
        Returns the URL for the given DIRECT_TEMPLATE name.
        Favors ${DIRECT_TEMPLATE}_SAVE_AS over the default path.
        :param name: name of the direct template
        :return: str
        """
        url = self.pelican_settings.get('{}_SAVE_AS'.format(name.upper()))
        if url is None:
            url = self.settings.get('{}_URL'.format(name.upper()), '{}.html'.format(name))
        return urljoin(self.url_site, url)