def process(self, html, url=None, smart_tidy=True):
        u""" Process HTML content or URL.

        For automatic extraction patterns and cleanups, :mod:`readability-lxml`
        is used, to stick as much as possible to the original PHP
        implementation and produce at least similar results with the same
        site config on the same article/content.

        :param html: an unicode string containing a full HTML page content.
            Expected to have a ``DOCTYPE`` and all other standard
            attributes ; eg. HTML fragments are not supported.
            It will be replaced, tidied, cleaned, striped, and all
            metadata and body attributes will be extracted from it.
            Beware : this HTML piece will be mauled. See source code for
            exact processing workflow, it's quite gorgeous.
        :type html: unicode

        :param url: as of version 0.5, this parameter is ignored. (**TODO**)
        :type url: str, unicode or ``None``

        :param smart_tidy: When ``True`` (default), runs :mod:`pytidylib`
            to tidy the HTML, after after run ``find_string``/``replace_string``
            replacements and before running extractions.
        :type smart_tidy: bool

        :returns: ``True`` on success, ``False`` on failure.
        :raises:
            - :class:`RuntimeError` if config has not been set at
              instantiation. This should change in the future by looking
              up a config if an ``url`` is passed as argument.

        .. note:: If tidy is used and no result is produced, we will try
            again without tidying.
            Generally speaking, tidy helps us deal with PHP's patchy HTML
            parsing (LOOOOOL. Zeriously?) most of the time but it has
            problems of its own which we try to avoid with this option.
            In the Python implementation, `pytidylib` has showed to help
            sanitize a lot the HTML before processing it. But nobody's
            perfect, and errors can happen in the Python world too, thus
            the *tidy* behavior was thought sane enough to be keep.
        """

        # TODO: re-implement URL handling with self.reset() here.

        if self.config is None:
            raise RuntimeError(u'extractor site config is not set.')

        # TODO: If re-running ourselves over an already-replaced string,
        #       this should just do nothing because everything has been
        #       done. We should have a test for that.
        html = self._process_replacements(html)

        # We keep the html untouched after replacements.
        # All processing happens on self.html after this point.
        self._tidy(html, smart_tidy)

        # return

        self._parse_html()

        self._extract_next_page_link()

        self._extract_title()

        self._extract_author()

        self._extract_language()

        self._extract_date()

        self._strip_unwanted_elements()

        self._extract_body()

        # TODO: re-implement auto-detection here.
        # NOTE: hNews extractor was here.
        # NOTE: instapaper extractor was here.

        self._auto_extract_if_failed()

        if self.title is not None or self.body is not None \
            or bool(self.author) or self.date is not None \
                or self.language is not None:
            self.success = True

        # if we've had no success and we've used tidy, there's a chance
        # that tidy has messed up. So let's try again without tidy...
        if not self.success and self.tidied and smart_tidy:
            self.process(html, url=None, smart_tidy=False)

        return self.success