def extract_tonnikala(fileobj, keywords, comment_tags, options):
    """Extract messages from Tonnikala files.

    :param fileobj: the file-like object the messages should be extracted
                    from
    :param keywords: a list of keywords (i.e. function names) that should
                     be recognized as translation functions
    :param comment_tags: a list of translator tags to search for and
                         include in the results
    :param options: a dictionary of additional options (optional)
    :return: an iterator over ``(lineno, funcname, message, comments)``
             tuples
    :rtype: ``iterator``
    """
    extractor = TonnikalaExtractor()
    for msg in extractor(filename=None, fileobj=fileobj, options=Options()):
        msgid = msg.msgid,

        prefix = ''
        if msg.msgid_plural:
            msgid = (msg.msgid_plural,) + msgid
            prefix = 'n'

        if msg.msgctxt:
            msgid = (msg.msgctxt,) + msgid
            prefix += 'p'

        yield (msg.location[1], prefix + 'gettext', msgid, msg.comment)