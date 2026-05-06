def hotp(key,counter,format='dec6',hash=hashlib.sha1):
    '''
       Compute a HOTP value as prescribed by RFC4226

       :param key:
           the HOTP secret key given as an hexadecimal string
       :param counter:
           the OTP generation counter
       :param format:
           the output format, can be:
              - hex, for a variable length hexadecimal format,
              - hex-notrunc, for a 40 characters hexadecimal non-truncated format,
              - dec4, for a 4 characters decimal format,
              - dec6,
              - dec7, or
              - dec8
           it defaults to dec6.
       :param hash:
           the hash module (usually from the hashlib package) to use,
           it defaults to hashlib.sha1.

       :returns:
           a string representation of the OTP value (as instructed by the format parameter).

       Examples:

        >>> hotp('343434', 2, format='dec6')
            '791903'
    '''
    bin_hotp = __hotp(key, counter, hash)

    if format == 'dec4':
        return dec(bin_hotp, 4)
    elif format == 'dec6':
        return dec(bin_hotp, 6)
    elif format == 'dec7':
        return dec(bin_hotp, 7)
    elif format == 'dec8':
        return dec(bin_hotp, 8)
    elif format == 'hex':
        return '%x' % truncated_value(bin_hotp)
    elif format == 'hex-notrunc':
        return _utils.tohex(bin_hotp)
    elif format == 'bin':
        return bin_hotp
    elif format == 'dec':
        return str(truncated_value(bin_hotp))
    else:
        raise ValueError('unknown format')