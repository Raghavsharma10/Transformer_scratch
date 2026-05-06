def extract(self, check_url=None, http_equiv_refresh=True):
        """
        Downloads HTML <head> tag first, extracts data from it using
        specific head techniques, loads it and checks if is complete. 
        Otherwise downloads the HTML <body> tag as well and loads data 
        extracted by using appropriate semantic techniques.

        Eagerly calls check_url(url) if any, before parsing the HTML.
        Provided function should raise an exception to break extraction.
        E.g.: URL has been summarized before; URL points to off limits
        websites like foursquare.com, facebook.com, bitly.com and so on.
        """
        # assert self._is_clear()
        logger = logging.getLogger(__name__)
        logger.info("Extract: %s", self.clean_url)
        with closing(request.get(self.clean_url, stream=True)) as response:
            response.raise_for_status()
            mime = response.headers.get('content-type')
            if mime and not ('html' in mime.lower()):
                raise HTMLParseError('Invalid Content-Type: %s' % mime)
            self.clean_url = self._clean_url(response.url)
            if self.clean_url is None:
                raise URLError('Bad url: %s' % response.url)
            if check_url is not None:
                check_url(url=self.clean_url)

            encoding = config.ENCODING or response.encoding

            self._html = ""
            if config.PHANTOMJS_BIN and \
                site(self.clean_url) in config.PHANTOMJS_SITES:
                self._html = request.phantomjs_get(self.clean_url)
                response.consumed = True

            head = self._get_tag(response, tag_name="head", encoding=encoding)

            if http_equiv_refresh:
                # Check meta http-equiv refresh tag
                html = head or decode(self._html, encoding)
                self._extract(html, self.clean_url, [
                    "summary.techniques.HTTPEquivRefreshTags",
                ])
                new_url = self.urls and self.urls[0]
                if new_url and new_url != self.clean_url:
                    logger.warning("Refresh: %s", new_url)
                    self._clear()
                    self.clean_url = new_url
                    return self.extract(check_url=check_url, http_equiv_refresh=False)

            if head:
                logger.debug("Got head: %s", len(head))
                self._extract(head, self.clean_url, [
                    "extraction.techniques.FacebookOpengraphTags",
                    "extraction.techniques.TwitterSummaryCardTags",
                    "extraction.techniques.HeadTags"
                ])
            else:
                logger.debug("No head: %s", self.clean_url)

            if config.GET_ALL_DATA or not self._is_complete():
                body = self._get_tag(response, tag_name="body", encoding=encoding)
                if body:
                    logger.debug("Got body: %s", len(body))
                    self._extract(body, self.clean_url, [
                        "extraction.techniques.HTML5SemanticTags",
                        "extraction.techniques.SemanticTags"                
                    ])
                else:
                    logger.debug("No body: %s", self.clean_url)

            if not head and not body:
                raise HTMLParseError('No head nor body tags found.')

            del self._html