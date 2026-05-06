def lookup(sock, domain, cache = None):
    """lookup an I2P domain name, returning a Destination instance"""
    domain = normalize_domain(domain)

    # cache miss, perform lookup
    reply = sam_cmd(sock, "NAMING LOOKUP NAME=%s" % domain)

    b64_dest = reply.get('VALUE')
    if b64_dest:
        dest = Dest(b64_dest, encoding='base64')
        if cache:
            cache[dest.base32 + '.b32.i2p'] = dest
        return dest
    else:
        raise NSError('Domain name %r not resolved because %r' % (domain, reply))