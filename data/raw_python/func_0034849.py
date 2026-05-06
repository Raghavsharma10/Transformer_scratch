def display(result, stream):
    """
    Intelligently print the result (or pass if result is None).

    :param result:
    :return: None
    """
    if result is None:
        return
    elif isinstance(result, basestring):
        pass
    elif isinstance(result, collections.Mapping):
        result = u'\n'.join(u'%s=%s' % (k, v) for
                            k, v in result.iteritems() if v is not None)
    elif isinstance(result, collections.Iterable):
        result = u'\n'.join(unicode(x) for x in result if x is not None)
    else:
        result = unicode(result)
    stream.write(result.encode('utf8'))
    stream.write('\n')