def uncomment(comment):
    """
    Converts the comment node received to a non-commented element, in place,
    and will return the new node.

    This may fail, primarily due to special characters within the comment that
    the xml parser is unable to handle. If it fails, this method will log an
    error and return None
    """
    parent = comment.parentNode
    h = html.parser.HTMLParser()
    data = h.unescape(comment.data)
    try:
        node = minidom.parseString(data).firstChild
    except xml.parsers.expat.ExpatError:  # Could not parse!
        log.error('Could not uncomment node due to parsing error!')
        return None
    else:
        parent.replaceChild(node, comment)
        return node