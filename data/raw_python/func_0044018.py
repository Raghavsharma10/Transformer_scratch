def reset(self):
        """ (re)set all instance attributes to default.

        Every attribute is set to ``None``, except :attr:`author`
        and :attr:`failures` which are set to ``[]``.
        """

        self.config = None
        self.html = None
        self.parsed_tree = None
        self.tidied = False
        self.next_page_link = None
        self.title = None
        self.author = set()
        self.language = None
        self.date = None
        self.body = None
        self.failures = set()
        self.success = False

        LOGGER.debug(u'Reset extractor instance to defaults/empty.')