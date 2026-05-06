def encode_sid(cls, secret, sid):
        """Computes the HMAC for the given session id."""

        secret_bytes = secret.encode("utf-8")
        sid_bytes = sid.encode("utf-8")

        sig = hmac.new(secret_bytes, sid_bytes, hashlib.sha512).hexdigest()
        return "%s%s" % (sig, sid)