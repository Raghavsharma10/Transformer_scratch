def request_sid(self):
        """ Request a BOSH session according to
        http://xmpp.org/extensions/xep-0124.html#session-request
        Returns the new SID (str).

        """
        if self._sid:
            return self._sid

        self.log.debug('Prepare to request BOSH session')

        data = self.send_request(self.get_body(sid_request=True))
        if not data:
            return None

        # This is XML. response_body contains the <body/> element of the
        # response.
        response_body = ET.fromstring(data)

        # Get the remote Session ID
        self._sid = response_body.get('sid')
        self.log.debug('sid = %s' % self._sid)

        # Get the longest time (s) that the XMPP server will wait before
        # responding to any request.
        self.server_wait = response_body.get('wait')
        self.log.debug('wait = %s' % self.server_wait)

        # Get the authid
        self.authid = response_body.get('authid')

        # Get the allowed authentication methods using xpath
        search_for = '{{{0}}}features/{{{1}}}mechanisms/{{{2}}}mechanism'.format(
            JABBER_STREAMS_NS, XMPP_SASL_NS, XMPP_SASL_NS
        )
        self.log.debug('Looking for "%s" into response body', search_for)
        mechanisms = response_body.findall(search_for)
        self.server_auth = []

        for mechanism in mechanisms:
            self.server_auth.append(mechanism.text)
            self.log.debug('New AUTH method: %s' % mechanism.text)

        if not self.server_auth:
            self.log.debug(('The server didn\'t send the allowed '
                            'authentication methods'))
            self._sid = None

        return self._sid