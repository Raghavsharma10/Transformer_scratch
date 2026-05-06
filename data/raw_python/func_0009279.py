def calculate_mac(mac_type, resource, content_hash):
    """Calculates a message authorization code (MAC)."""
    normalized = normalize_string(mac_type, resource, content_hash)
    log.debug(u'normalized resource for mac calc: {norm}'
              .format(norm=normalized))
    digestmod = getattr(hashlib, resource.credentials['algorithm'])

    # Make sure we are about to hash binary strings.

    if not isinstance(normalized, six.binary_type):
        normalized = normalized.encode('utf8')
    key = resource.credentials['key']
    if not isinstance(key, six.binary_type):
        key = key.encode('ascii')

    result = hmac.new(key, normalized, digestmod)
    return b64encode(result.digest())