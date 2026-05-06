def calculate_payload_hash(payload, algorithm, content_type):
    """Calculates a hash for a given payload."""
    p_hash = hashlib.new(algorithm)

    parts = []
    parts.append('hawk.' + str(HAWK_VER) + '.payload\n')
    parts.append(parse_content_type(content_type) + '\n')
    parts.append(payload or '')
    parts.append('\n')

    for i, p in enumerate(parts):
        # Make sure we are about to hash binary strings.
        if not isinstance(p, six.binary_type):
            p = p.encode('utf8')
        p_hash.update(p)
        parts[i] = p

    log.debug('calculating payload hash from:\n{parts}'
              .format(parts=pprint.pformat(parts)))

    return b64encode(p_hash.digest())