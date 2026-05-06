def calculate_ts_mac(ts, credentials):
    """Calculates a message authorization code (MAC) for a timestamp."""
    normalized = ('hawk.{hawk_ver}.ts\n{ts}\n'
                  .format(hawk_ver=HAWK_VER, ts=ts))
    log.debug(u'normalized resource for ts mac calc: {norm}'
              .format(norm=normalized))
    digestmod = getattr(hashlib, credentials['algorithm'])

    if not isinstance(normalized, six.binary_type):
        normalized = normalized.encode('utf8')
    key = credentials['key']
    if not isinstance(key, six.binary_type):
        key = key.encode('ascii')

    result = hmac.new(key, normalized, digestmod)
    return b64encode(result.digest())