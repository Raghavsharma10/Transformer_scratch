def parse_text(infile, xpath=None, filter_words=None, attributes=None):
    """Filter text using XPath, regex keywords, and tag attributes.

    Keyword arguments:
    infile -- HTML or text content to parse (list)
    xpath -- an XPath expression (str)
    filter_words -- regex keywords (list)
    attributes -- HTML tag attributes (list)

    Return a list of strings of text.
    """
    infiles = []
    text = []
    if xpath is not None:
        infile = parse_html(infile, xpath)
        if isinstance(infile, list):
            if isinstance(infile[0], lh.HtmlElement):
                infiles = list(infile)
            else:
                text = [line + '\n' for line in infile]
        elif isinstance(infile, lh.HtmlElement):
            infiles = [infile]
        else:
            text = [infile]
    else:
        infiles = [infile]

    if attributes is not None:
        attributes = [clean_attr(x) for x in attributes]
        attributes = [x for x in attributes if x]
    else:
        attributes = ['text()']

    if not text:
        text_xpath = '//*[not(self::script) and not(self::style)]'
        for attr in attributes:
            for infile in infiles:
                if isinstance(infile, lh.HtmlElement):
                    new_text = infile.xpath('{0}/{1}'.format(text_xpath, attr))
                else:
                    # re.split preserves delimiters place in the list
                    new_text = [x for x in re.split('(\n)', infile) if x]
                text += new_text

    if filter_words is not None:
        text = re_filter(text, filter_words)
    return [''.join(x for x in line if x in string.printable)
            for line in remove_whitespace(text) if line]