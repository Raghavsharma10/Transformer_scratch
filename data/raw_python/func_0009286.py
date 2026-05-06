def check_bewit(url, credential_lookup, now=None):
    """
    Validates the given bewit.

    Returns True if the resource has a valid bewit parameter attached,
    or raises a subclass of HawkFail otherwise.

    :param credential_lookup:
        Callable to look up the credentials dict by sender ID.
        The credentials dict must have the keys:
        ``id``, ``key``, and ``algorithm``.
        See :ref:`receiving-request` for an example.
    :type credential_lookup: callable

    :param now=None:
        Unix epoch time for the current time to determine if bewit has expired.
        If None, then the current time as given by utc_now() is used.
    :type now=None: integer
    """
    raw_bewit, stripped_url = strip_bewit(url)
    bewit = parse_bewit(raw_bewit)
    try:
        credentials = credential_lookup(bewit.id)
    except LookupError:
        raise CredentialsLookupError('Could not find credentials for ID {0}'
                                     .format(bewit.id))

    res = Resource(url=stripped_url,
                   method='GET',
                   credentials=credentials,
                   timestamp=bewit.expiration,
                   nonce='',
                   ext=bewit.ext,
                   )
    mac = calculate_mac('bewit', res, None)
    mac = mac.decode('ascii')

    if not strings_match(mac, bewit.mac):
        raise MacMismatch('bewit with mac {bewit_mac} did not match expected mac {expected_mac}'
                          .format(bewit_mac=bewit.mac,
                                  expected_mac=mac))

    # Check that the timestamp isn't expired
    if now is None:
        # TODO: Add offset/skew
        now = utc_now()
    if int(bewit.expiration) < now:
        # TODO: Refactor TokenExpired to handle this better
        raise TokenExpired('bewit with UTC timestamp {ts} has expired; '
                           'it was compared to {now}'
                           .format(ts=bewit.expiration, now=now),
                           localtime_in_seconds=now,
                           www_authenticate=''
                           )

    return True