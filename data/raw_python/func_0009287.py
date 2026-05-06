def accept_response(self,
                        response_header,
                        content=EmptyValue,
                        content_type=EmptyValue,
                        accept_untrusted_content=False,
                        localtime_offset_in_seconds=0,
                        timestamp_skew_in_seconds=default_ts_skew_in_seconds,
                        **auth_kw):
        """
        Accept a response to this request.

        :param response_header:
            A `Hawk`_ ``Server-Authorization`` header
            such as one created by :class:`mohawk.Receiver`.
        :type response_header: str

        :param content=EmptyValue: Byte string of the response body received.
        :type content=EmptyValue: str

        :param content_type=EmptyValue:
            Content-Type header value of the response received.
        :type content_type=EmptyValue: str

        :param accept_untrusted_content=False:
            When True, allow responses that do not hash their content.
            Read :ref:`skipping-content-checks` to learn more.
        :type accept_untrusted_content=False: bool

        :param localtime_offset_in_seconds=0:
            Seconds to add to local time in case it's out of sync.
        :type localtime_offset_in_seconds=0: float

        :param timestamp_skew_in_seconds=60:
            Max seconds until a message expires. Upon expiry,
            :class:`mohawk.exc.TokenExpired` is raised.
        :type timestamp_skew_in_seconds=60: float

        .. _`Hawk`: https://github.com/hueniverse/hawk
        """
        log.debug('accepting response {header}'
                  .format(header=response_header))

        parsed_header = parse_authorization_header(response_header)

        resource = Resource(ext=parsed_header.get('ext', None),
                            content=content,
                            content_type=content_type,
                            # The following response attributes are
                            # in reference to the original request,
                            # not to the reponse header:
                            timestamp=self.req_resource.timestamp,
                            nonce=self.req_resource.nonce,
                            url=self.req_resource.url,
                            method=self.req_resource.method,
                            app=self.req_resource.app,
                            dlg=self.req_resource.dlg,
                            credentials=self.credentials,
                            seen_nonce=self.seen_nonce)

        self._authorize(
            'response', parsed_header, resource,
            # Per Node lib, a responder macs the *sender's* timestamp.
            # It does not create its own timestamp.
            # I suppose a slow response could time out here. Maybe only check
            # mac failures, not timeouts?
            their_timestamp=resource.timestamp,
            timestamp_skew_in_seconds=timestamp_skew_in_seconds,
            localtime_offset_in_seconds=localtime_offset_in_seconds,
            accept_untrusted_content=accept_untrusted_content,
            **auth_kw)