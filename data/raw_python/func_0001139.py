def make_msgid(idstring=None, utc=False):
    """Return a string suitable for RFC 2822 compliant Message-ID.

    E.g: <20020201195627.33539.96671@nightshade.la.mastaler.com>

    Optional idstring if given is a string used to strengthen the
    uniqueness of the message id.
    """
    if utc:
        timestamp = time.gmtime()
    else:
        timestamp = time.localtime()
    utcdate = time.strftime("%Y%m%d%H%M%S", timestamp)
    try:
        pid = os.getpid()
    except AttributeError:
        # No getpid() in Jython, for example.
        pid = 1
    randint = random.randrange(100000)
    if idstring is None:
        idstring = ""
    else:
        idstring = "." + idstring
    idhost = DNS_NAME
    msgid = "<%s.%s.%s%s@%s>" % (utcdate, pid, randint, idstring, idhost)
    return msgid