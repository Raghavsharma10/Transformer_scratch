def hash(password, salt, N, r, p, dkLen):
    """Returns the result of the scrypt password-based key derivation function.

       Constraints:
         r * p < (2 ** 30)
         dkLen <= (((2 ** 32) - 1) * 32
         N must be a power of 2 greater than 1 (eg. 2, 4, 8, 16, 32...)
         N, r, p must be positive
     """

    # This only matters to Python 3
    if not check_bytes(password):
        raise ValueError('password must be a byte array')

    if not check_bytes(salt):
        raise ValueError('salt must be a byte array')

    # Scrypt implementation. Significant thanks to https://github.com/wg/scrypt
    if N < 2 or (N & (N - 1)): raise ValueError('Scrypt N must be a power of 2 greater than 1')

    # A psuedorandom function
    prf = lambda k, m: hmac.new(key = k, msg = m, digestmod = hashlib.sha256).digest()

    # convert into integers
    B  = [ get_byte(c) for c in pbkdf2_single(password, salt, p * 128 * r, prf) ]
    B = [ ((B[i + 3] << 24) | (B[i + 2] << 16) | (B[i + 1] << 8) | B[i + 0]) for i in xrange(0, len(B), 4)]

    XY = [ 0 ] * (64 * r)
    V  = [ 0 ] * (32 * r * N)

    for i in xrange(0, p):
        smix(B, i * 32 * r, r, N, V, XY)

    # Convert back into bytes
    Bc = [ ]
    for i in B:
        Bc.append((i >> 0) & 0xff)
        Bc.append((i >> 8) & 0xff)
        Bc.append((i >> 16) & 0xff)
        Bc.append((i >> 24) & 0xff)

    return pbkdf2_single(password, chars_to_bytes(Bc), dkLen, prf)