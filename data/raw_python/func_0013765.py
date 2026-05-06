def from_b32key(b32_key, state=None):
    '''Some phone app directly accept a partial b32 encoding, we try to emulate that'''
    try:
        lenient_b32decode(b32_key)
    except TypeError:
        raise ValueError('invalid base32 value')
    return GoogleAuthenticator('otpauth://totp/xxx?%s' %
            urlencode({'secret': b32_key}), state=state)