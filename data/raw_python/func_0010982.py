def get_search_cache_key(prefix, *args):
    """ Generate suitable key to cache twitter tag context
    """
    key = '%s_%s' % (prefix, '_'.join([str(arg) for arg in args if arg]))
    not_allowed = re.compile('[^%s]' % ''.join([chr(i) for i in range(33, 128)]))
    key = not_allowed.sub('', key)
    return key