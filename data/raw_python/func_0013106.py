def urlsafe(self):
    """Return a url-safe string encoding this Key's Reference.

    This string is compatible with other APIs and languages and with
    the strings used to represent Keys in GQL and in the App Engine
    Admin Console.
    """
    # This is 3-4x faster than urlsafe_b64decode()
    urlsafe = base64.b64encode(self.reference().Encode())
    return urlsafe.rstrip('=').replace('+', '-').replace('/', '_')