def base64url_decode(msg):
    """
    Decode a base64 message based on JWT spec, Appendix B.
    "Notes on implementing base64url encoding without padding"
    """
    rem = len(msg) % 4
    if rem:
        msg += b'=' * (4 - rem)

    return base64.urlsafe_b64decode(msg)