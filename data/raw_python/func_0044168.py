def reset(self):
        """ (re)set all attributes to defaults (eg. empty sets or ``None``). """

        # Use first matching element as title (0 or more xpath expressions)
        self.title = OrderedSet()

        # Use first matching element as body (0 or more xpath expressions)
        self.body = OrderedSet()

        # Use first matching element as author (0 or more xpath expressions)
        self.author = OrderedSet()

        # Use first matching element as date (0 or more xpath expressions)
        self.date = OrderedSet()

        # Put language here. It's not supported in siteconfig syntax,
        # but having it here allows more generic handling in extractor.
        self.language = (
            '//html[@lang]/@lang',
            '//meta[@name="DC.language"]/@content',
        )

        # Strip elements matching these xpath expressions (0 or more)
        self.strip = OrderedSet()

        # Strip 0 or more elements which contain these
        # strings in the id or class attribute.
        self.strip_id_or_class = OrderedSet()

        # Strip 0 or more images which contain
        # these strings in the src attribute.
        self.strip_image_src = OrderedSet()

        # Additional HTTP headers to send
        # NOT YET USED
        self.http_header = OrderedSet()

        # For those 3, None means that default will be used. But we need
        # None to distinguish from False during multiple configurations
        # merges.
        self.tidy = None
        self.prune = None
        self.autodetect_on_failure = None

        # Test URL - if present, can be used to test the config above
        self.test_url = OrderedSet()
        self.test_contains = OrderedSet()

        # Single-page link should identify a link element or URL pointing
        # to the page holding the entire article.
        #
        # This is useful for sites which split their articles across
        # multiple pages. Links to such pages tend to display the first
        # page with links to the other pages at the bottom.
        #
        # Often there is also a link to a page which displays the entire
        # article on one page (e.g. 'print view').
        #
        # `single_page_link` should be an XPath expression identifying the
        # link to that single page. If present and we find a match, we will
        # retrieve that page and the rest of the options in this config will
        # be applied to the new page.
        self.single_page_link = OrderedSet()

        self.next_page_link = OrderedSet()

        # Single-page link in feed? - same as above, but patterns applied
        # to item description HTML taken from feed. XXX
        self.single_page_link_in_feed = OrderedSet()

        # Which parser to use for turning raw HTML into a DOMDocument,
        # either `libxml` (PHP) / `lxml` (Python) or `html5lib`. Defaults
        # to `lxml` if None.
        self.parser = None

        # Strings to search for in HTML before processing begins. Goes by
        # pairs with `replace_string`. Not a set because we can have more
        # than one of the same, to be replaced by different values.
        self.find_string = []

        # Strings to replace those found in `find_string` before HTML
        # processing begins.
        self.replace_string = []