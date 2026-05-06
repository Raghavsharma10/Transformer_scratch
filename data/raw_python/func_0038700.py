def SessionAuthSourceInitializer(
    value_key='sanity.'
):
    """ An authentication source that uses the current session """

    value_key = value_key + 'value'

    @implementer(IAuthSourceService)
    class SessionAuthSource(object):
        vary = []

        def __init__(self, context, request):
            self.request = request
            self.session = request.session
            self.cur_val = None

        def get_value(self):
            if self.cur_val is None:
                self.cur_val = self.session.get(value_key, [None, None])

            return self.cur_val

        def headers_remember(self, value):
            if self.cur_val is None:
                self.cur_val = self.session.get(value_key, [None, None])

            self.session[value_key] = value
            return []

        def headers_forget(self):
            if self.cur_val is None:
                self.cur_val = self.session.get(value_key, [None, None])

            if value_key in self.session:
                del self.session[value_key]
            return []

    return SessionAuthSource