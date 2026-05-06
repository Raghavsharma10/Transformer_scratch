def respond(self,
                content=EmptyValue,
                content_type=EmptyValue,
                always_hash_content=True,
                ext=None):
        """
        Respond to the request.

        This generates the :attr:`mohawk.Receiver.response_header`
        attribute.

        :param content=EmptyValue: Byte string of response body that will be sent.
        :type content=EmptyValue: str

        :param content_type=EmptyValue: content-type header value for response.
        :type content_type=EmptyValue: str

        :param always_hash_content=True:
            When True, ``content`` and ``content_type`` must be provided.
            Read :ref:`skipping-content-checks` to learn more.
        :type always_hash_content=True: bool

        :param ext=None:
            An external `Hawk`_ string. If not None, this value will be
            signed so that the sender can trust it.
        :type ext=None: str

        .. _`Hawk`: https://github.com/hueniverse/hawk
        """

        log.debug('generating response header')

        resource = Resource(url=self.resource.url,
                            credentials=self.resource.credentials,
                            ext=ext,
                            app=self.parsed_header.get('app', None),
                            dlg=self.parsed_header.get('dlg', None),
                            method=self.resource.method,
                            content=content,
                            content_type=content_type,
                            always_hash_content=always_hash_content,
                            nonce=self.parsed_header['nonce'],
                            timestamp=self.parsed_header['ts'])

        mac = calculate_mac('response', resource, resource.gen_content_hash())

        self.response_header = self._make_header(resource, mac,
                                                 additional_keys=['ext'])
        return self.response_header