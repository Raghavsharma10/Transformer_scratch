def xpath(self, selector: str, *, first: bool = False, _encoding: str = None) -> _XPath:
        """Given an XPath selector, returns a list of
        :class:`Element <Element>` objects or a single one.

        :param selector: XPath Selector to use.
        :param first: Whether or not to return just the first result.
        :param _encoding: The encoding format.

        If a sub-selector is specified (e.g. ``//a/@href``), a simple
        list of results is returned.

        See W3School's `XPath Examples
        <https://www.w3schools.com/xml/xpath_examples.asp>`_
        for more details.

        If ``first`` is ``True``, only returns the first
        :class:`Element <Element>` found.
        """
        selected = self.lxml.xpath(selector)

        elements = [
            Element(element=selection, default_encoding=_encoding or self.encoding)
            if not isinstance(selection, etree._ElementUnicodeResult) else str(selection)
            for selection in selected
        ]

        return _get_first_or_list(elements, first)