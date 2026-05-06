def HeaderAuthSourceInitializer(
    secret,
    salt='sanity.header.'
):
    """ An authentication source that uses the Authorization header. """

    @implementer(IAuthSourceService)
    class HeaderAuthSource(object):
        vary = ['Authorization']

        def __init__(self, context, request):
            self.request = request
            self.cur_val = None

            serializer = JSONSerializer()
            self.serializer = SignedSerializer(
                secret,
                salt,
                serializer=serializer,
                )

        def _get_authorization(self):
            try:
                type, token = self.request.authorization

                return self.serializer.loads(token)
            except Exception:
                return None

        def _create_authorization(self, value):
            try:
                return self.serializer.dumps(value)
            except Exception:
                return ''

        def get_value(self):
            if self.cur_val is None:
                self.cur_val = self._get_authorization() or [None, None]

            return self.cur_val

        def headers_remember(self, value):
            if self.cur_val is None:
                self.cur_val = None

            token = self._create_authorization(value)
            auth_info = native_(b'Bearer ' + token, 'latin-1', 'strict')
            return [('Authorization', auth_info)]

        def headers_forget(self):
            if self.cur_val is None:
                self.cur_val = None

            return []

    return HeaderAuthSource