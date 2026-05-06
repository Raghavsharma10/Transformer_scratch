def _extract_language(self):
        """ Extract language from the HTML ``<head>`` tags. """

        if self.language:
            return

        found = False

        for pattern in self.config.language:
            for item in self.parsed_tree.xpath(pattern):
                stripped_language = item.strip()

                if stripped_language:
                    self.language = stripped_language
                    LOGGER.info(u'Language extracted: %s.', stripped_language,
                                extra={'siteconfig': self.config.host})
                    found = True
                    break

            if found:
                break