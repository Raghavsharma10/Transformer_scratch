def parse_headers(content_disposition, location=None, relaxed=False):
    """Build a ContentDisposition from header values.
    """

    LOGGER.debug(
        'Content-Disposition %r, Location %r', content_disposition, location)

    if content_disposition is None:
        return ContentDisposition(location=location)

    # Both alternatives seem valid.
    if False:
        # Require content_disposition to be ascii bytes (0-127),
        # or characters in the ascii range
        content_disposition = ensure_charset(content_disposition, 'ascii')
    else:
        # We allow non-ascii here (it will only be parsed inside of
        # qdtext, and rejected by the grammar if it appears in
        # other places), although parsing it can be ambiguous.
        # Parsing it ensures that a non-ambiguous filename* value
        # won't get dismissed because of an unrelated ambiguity
        # in the filename parameter. But it does mean we occasionally
        # give less-than-certain values for some legacy senders.
        content_disposition = ensure_charset(content_disposition, 'iso-8859-1')

    # Check the caller already did LWS-folding (normally done
    # when separating header names and values; RFC 2616 section 2.2
    # says it should be done before interpretation at any rate).
    # Hopefully space still means what it should in iso-8859-1.
    # This check is a bit stronger that LWS folding, it will
    # remove CR and LF even if they aren't part of a CRLF.
    # However http doesn't allow isolated CR and LF in headers outside
    # of LWS.

    if relaxed:
        # Relaxed has two effects (so far):
        # the grammar allows a final ';' in the header;
        # we do LWS-folding, and possibly normalise other broken
        # whitespace, instead of rejecting non-lws-safe text.
        # XXX Would prefer to accept only the quoted whitespace
        # case, rather than normalising everything.
        content_disposition = normalize_ws(content_disposition)
        parser = content_disposition_value_relaxed
    else:
        # Turns out this is occasionally broken: two spaces inside
        # a quoted_string's qdtext. Firefox and Chrome save the two spaces.
        if not is_lws_safe(content_disposition):
            raise ValueError(
                content_disposition, 'Contains nonstandard whitespace')

        parser = content_disposition_value

    try:
        parsed = parser.parse(content_disposition)
    except FullFirstMatchException:
        return ContentDisposition(location=location)
    return ContentDisposition(
        disposition=parsed[0], assocs=parsed[1:], location=location)