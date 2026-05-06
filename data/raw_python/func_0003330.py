def cookietostr(self):
        "Cookie values are bytes in Python3. This function Convert bytes to string with env.encoding(default to utf-8)."
        self.cookies = dict((k, (v.decode(self.encoding) if not isinstance(v, str) else v)) for k,v in self.cookies.items())
        return self.cookies