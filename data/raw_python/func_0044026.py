def _auto_extract_if_failed(self):
        """ Try to automatically extract as much as possible. """

        if not self.config.autodetect_on_failure:
            return

        readabilitized = Document(self.html)

        if self.title is None:
            if bool(self.config.title):
                self.failures.add('title')

            title = readabilitized.title().strip()

            if title:
                self.title = title
                LOGGER.info(u'Title extracted in automatic mode.',
                            extra={'siteconfig': self.config.host})

            else:
                self.failures.add('title')

        if self.body is None:
            if bool(self.config.body):
                self.failures.add('body')

            body = readabilitized.summary().strip()

            if body:
                self.body = body
                LOGGER.info(u'Body extracted in automatic mode.',
                            extra={'siteconfig': self.config.host})

            else:
                self.failures.add('body')

        for attr_name in ('date', 'language', 'author', ):
            if not bool(getattr(self, attr_name, None)):
                if bool(getattr(self.config, attr_name, None)):
                    self.failures.add(attr_name)
                    LOGGER.warning(u'Could not extract any %s from XPath '
                                   u'expression(s) %s.', attr_name,
                                   u', '.join(getattr(self.config, attr_name)),
                                   extra={'siteconfig': self.config.host})