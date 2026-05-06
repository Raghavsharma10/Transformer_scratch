def cookie(data, key_salt='', secret=None, digestmod=None):
    """ Encodes or decodes a signed cookie.
        @data: cookie data
        @key_salt: HMAC key signing salt
        @secret: HMAC signing secret key
        @digestmod: hashing algorithm to sign with, recommended >=sha256

        -> HMAC signed or unsigned cookie data

        ..
            from vital.security import cookie

            cookie("Hello, world.", "saltyDog", secret="alBVlwe")
            # -> '!YuOoKwDp8GhrwwojdjTxSCj1c2Z+7yz7r6cC7E3hBWo=?IkhlbGxvLCB3b3JsZC4i'
            cookie(
                "!YuOoKwDp8GhrwwojdjTxSCj1c2Z+7yz7r6cC7E3hBWo=?IkhlbGxvLCB3b3JsZC4i",
                "saltyDog", secret="alBVlwe")
            # -> 'Hello, world.'
        ..
    """
    digestmod = digestmod or sha256
    if not data:
        return None
    try:
        # Decode signed cookie
        assert cookie_is_encoded(data)
        datab = uniorbytes(data, bytes)
        sig, msg = datab.split(uniorbytes('?', bytes), 1)
        key = ("{}{}").format(secret, key_salt)
        sig_check = hmac.new(
            key=uniorbytes(key, bytes), msg=msg, digestmod=digestmod).digest()
        sig_check = uniorbytes(b64encode(sig_check), bytes)
        if lscmp(sig[1:], sig_check):
            return json.loads(uniorbytes(b64decode(msg)))
        return None
    except:
        # Encode and sign a json-able object. Return a string.
        key = ("{}{}").format(secret, key_salt)
        msg = b64encode(uniorbytes(json.dumps(data), bytes))
        sig = hmac.new(
            key=uniorbytes(key, bytes), msg=msg,
            digestmod=digestmod).digest()
        sig = uniorbytes(b64encode(sig), bytes)
        return uniorbytes('!'.encode() + sig + '?'.encode() + msg)