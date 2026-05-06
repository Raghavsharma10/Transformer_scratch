def _process_replacements(self, html):
        """ Do raw string replacements on :param:`html`. """

        if self.config.find_string:
            for find_pattern, replace_pattern in self.config.replace_patterns:
                html = html.replace(find_pattern, replace_pattern)

            LOGGER.info(u'Done replacements.',
                        extra={'siteconfig': self.config.host})

        return html