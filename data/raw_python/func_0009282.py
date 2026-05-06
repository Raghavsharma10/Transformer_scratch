def parse_authorization_header(auth_header):
    """
    Example Authorization header:

        'Hawk id="dh37fgj492je", ts="1367076201", nonce="NPHgnG", ext="and
        welcome!", mac="CeWHy4d9kbLGhDlkyw2Nh3PJ7SDOdZDa267KH4ZaNMY="'
    """
    if len(auth_header) > MAX_LENGTH:
        raise BadHeaderValue('Header exceeds maximum length of {max_length}'.format(
            max_length=MAX_LENGTH))

    # Make sure we have a unicode object for consistency.
    if isinstance(auth_header, six.binary_type):
        auth_header = auth_header.decode('utf8')

    scheme, attributes_string = auth_header.split(' ', 1)

    if scheme.lower() != 'hawk':
        raise HawkFail("Unknown scheme '{scheme}' when parsing header"
                       .format(scheme=scheme))


    attributes = {}

    def replace_attribute(match):
        """Extract the next key="value"-pair in the header."""
        key = match.group('key')
        value = match.group('value')
        if key not in allowable_header_keys:
            raise HawkFail("Unknown Hawk key '{key}' when parsing header"
                           .format(key=key))
        validate_header_attr(value, name=key)
        if key in attributes:
            raise BadHeaderValue('Duplicate key in header: {key}'.format(key=key))
        attributes[key] = value

    # Iterate over all the key="value"-pairs in the header, replace them with
    # an empty string, and store the extracted attribute in the attributes
    # dict. Correctly formed headers will then leave nothing unparsed ('').
    unparsed_header = HAWK_HEADER_RE.sub(replace_attribute, attributes_string)
    if unparsed_header != '':
        raise BadHeaderValue("Couldn't parse Hawk header", unparsed_header)

    log.debug('parsed Hawk header: {header} into: \n{parsed}'
              .format(header=auth_header, parsed=pprint.pformat(attributes)))
    return attributes