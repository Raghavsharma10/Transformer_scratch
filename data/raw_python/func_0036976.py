def read_unicode(path, encoding, encoding_errors):
    """ Return the contents of a file as a unicode string. """
    try:
      f = open(path, 'rb')
      return make_unicode(f.read(), encoding, encoding_errors)
    finally:
      f.close()