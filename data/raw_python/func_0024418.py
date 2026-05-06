def post(self, view_name):
        """
        login handler
        """
        sess_id = None
        input_data = {}
        # try:
        self._handle_headers()

        # handle input
        input_data = json_decode(self.request.body) if self.request.body else {}
        input_data['path'] = view_name

        # set or get session cookie
        if not self.get_cookie(COOKIE_NAME) or 'username' in input_data:
            sess_id = uuid4().hex
            self.set_cookie(COOKIE_NAME, sess_id)  # , domain='127.0.0.1'
        else:
            sess_id = self.get_cookie(COOKIE_NAME)
        # h_sess_id = "HTTP_%s" % sess_id
        input_data = {'data': input_data,
                      '_zops_remote_ip': self.request.remote_ip}
        log.info("New Request for %s: %s" % (sess_id, input_data))

        self.application.pc.register_websocket(sess_id, self)
        self.application.pc.redirect_incoming_message(sess_id,
                                                      json_encode(input_data),
                                                      self.request)