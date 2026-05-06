def decode_sid(cls, secret, cookie_value):
        """Decodes a cookie value and returns the sid if value or None if invalid."""
        if len(cookie_value) > SIG_LENGTH + SID_LENGTH:
            logging.warn("cookie value is incorrect length")
            return None

        cookie_sig = cookie_value[:SIG_LENGTH]
        cookie_sid = cookie_value[SIG_LENGTH:]

        secret_bytes = secret.encode("utf-8")
        cookie_sid_bytes = cookie_sid.encode("utf-8")

        actual_sig = hmac.new(secret_bytes, cookie_sid_bytes, hashlib.sha512).hexdigest()

        if not Session.is_signature_equal(cookie_sig, actual_sig):
            return None

        return cookie_sid