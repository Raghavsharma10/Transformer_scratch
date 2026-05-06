def totp(key, format='dec6', period=30, t=None, hash=hashlib.sha1):
    '''
       Compute a TOTP value as prescribed by OATH specifications.

       :param key:
           the TOTP key given as an hexadecimal string
       :param format:
           the output format, can be:
              - hex, for a variable length hexadecimal format,
              - hex-notrunc, for a 40 characters hexadecimal non-truncated format,
              - dec4, for a 4 characters decimal format,
              - dec6,
              - dec7, or
              - dec8
           it defaults to dec6.
       :param period:
           a positive integer giving the period between changes of the OTP
           value, as seconds, it defaults to 30.
       :param t:
           a positive integer giving the current time as seconds since EPOCH
           (1st January 1970 at 00:00 GMT), if None we use time.time(); it
           defaults to None;
       :param hash:
           the hash module (usually from the hashlib package) to use,
           it defaults to hashlib.sha1.

       :returns:
           a string representation of the OTP value (as instructed by the format parameter).
       :type: str
    '''
    if t is None:
        t = int(time.time())
    else:
        if isinstance(t, datetime.datetime):
            t = calendar.timegm(t.utctimetuple())
        else:
            t = int(t)
    T = int(t/period)
    return hotp(key, T, format=format, hash=hash)