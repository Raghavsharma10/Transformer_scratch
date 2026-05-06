def CookieAuthSourceInitializer(
    secret,
    cookie_name='auth',
    secure=False,
    max_age=None,
    httponly=False,
    path="/",
    domains=None,
    debug=False,
    hashalg='sha512',
):
    """ An authentication source that uses a unique cookie. """

    @implementer(IAuthSourceService)
    class CookieAuthSource(object):
        vary = ['Cookie']

        def __init__(self, context, request):
            self.domains = domains

            if self.domains is None:
                self.domains = []
                self.domains.append(request.domain)

            self.cookie = SignedCookieProfile(
                secret,
                'authsanity',
                cookie_name,
                secure=secure,
                max_age=max_age,
                httponly=httponly,
                path=path,
                domains=domains,
                hashalg=hashalg,
                )
            # Bind the cookie to the current request
            self.cookie = self.cookie.bind(request)

        def get_value(self):
            val = self.cookie.get_value()

            if val is None:
                return [None, None]

            return val

        def headers_remember(self, value):
            return self.cookie.get_headers(value, domains=self.domains)

        def headers_forget(self):
            return self.cookie.get_headers(None, max_age=0)

    return CookieAuthSource