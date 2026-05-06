def repercent_broken_unicode(path):
    """
    As per section 3.2 of RFC 3987, step three of converting a URI into an IRI,
    we need to re-percent-encode any octet produced that is not part of a
    strictly legal UTF-8 octet sequence.
    """
    # originally from django.utils.encoding
    while True:
        try:
            return path.decode('utf-8')
        except UnicodeDecodeError as e:
            repercent = quote(path[e.start:e.end], safe=b"/#%[]=:;$&()+,!?*@'~")
            path = path[:e.start] + repercent.encode('ascii') + path[e.end:]