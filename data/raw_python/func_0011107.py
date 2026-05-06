def stable_format_dict(d):
    """A sorted, python2/3 stable formatting of a dictionary.

    Does not work for dicts with unicode strings as values."""
    inner = ', '.join('{}: {}'.format(repr(k)[1:]
                                      if repr(k).startswith(u"u'") or repr(k).startswith(u'u"')
                                      else repr(k),
                                      v)
                      for k, v in sorted(d.items()))
    return '{%s}' % inner