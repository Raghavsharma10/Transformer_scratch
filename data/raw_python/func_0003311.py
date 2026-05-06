def setcookie(self, key, value, max_age=None, expires=None, path='/', domain=None, secure=None, httponly=False):
        """
        Add a new cookie
        """
        newcookie = Morsel()
        newcookie.key = key
        newcookie.value = value
        newcookie.coded_value = value
        if max_age is not None:
            newcookie['max-age'] = max_age
        if expires is not None:
            newcookie['expires'] = expires
        if path is not None:
            newcookie['path'] = path
        if domain is not None:
            newcookie['domain'] = domain
        if secure:
            newcookie['secure'] = secure
        if httponly:
            newcookie['httponly'] = httponly
        self.sent_cookies = [c for c in self.sent_cookies if c.key != key]
        self.sent_cookies.append(newcookie)