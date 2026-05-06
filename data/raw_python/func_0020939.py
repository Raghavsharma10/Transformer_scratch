def parse(document, clean_html=True, unix_timestamp=False, encoding=None):
    """Parse a document and return a feedparser dictionary with attr key access.
    If clean_html is False, the html in the feed will not be cleaned.  If
    clean_html is True, a sane version of lxml.html.clean.Cleaner will be used.
    If it is a Cleaner object, that cleaner will be used.  If unix_timestamp is
    True, the date information will be a numerical unix timestamp rather than a
    struct_time.  If encoding is provided, the encoding of the document will be
    manually set to that."""
    if isinstance(clean_html, bool):
        cleaner = default_cleaner if clean_html else fake_cleaner
    else:
        cleaner = clean_html
    result = feedparser.FeedParserDict()
    result['feed'] = feedparser.FeedParserDict()
    result['entries'] = []
    result['bozo'] = 0
    try:
        parser = SpeedParser(document, cleaner, unix_timestamp, encoding)
        parser.update(result)
    except Exception as e:
        if isinstance(e, UnicodeDecodeError) and encoding is True:
            encoding = chardet.detect(document)['encoding']
            document = document.decode(encoding, 'replace').encode('utf-8')
            return parse(document, clean_html, unix_timestamp, encoding)
        import traceback
        result['bozo'] = 1
        result['bozo_exception'] = e
        result['bozo_tb'] = traceback.format_exc()
    return result