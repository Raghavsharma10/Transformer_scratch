def _tidy(self, html, smart_tidy):
        """ Tidy HTML if we have a tidy method.

        This fixes problems with some sites which would otherwise trouble
        DOMDocument's HTML parsing.

        Although sometimes it makes the problem worse, which is why we can
        override it in site config files.
        """

        if self.config.tidy and tidylib and smart_tidy:

            try:
                document, errors = tidylib.tidy_document(html, self.tidy_config)

            except UnicodeDecodeError:
                # For some reason, pytidylib fails to decode, whereas the
                # original html content converts perfectly manually.
                document, errors = tidylib.tidy_document(html.encode('utf-8'),
                                                         self.tidy_config)
                document = document.decode('utf-8')
            # if errors:
            #     LOGGER.debug(u'Ignored errors returned by tidylib: %s',
            #                  errors)

            self.tidied = True
            self.html = document

            LOGGER.info(u'Tidied document.')

        else:
            self.html = html