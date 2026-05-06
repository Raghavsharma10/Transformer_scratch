def _credssp_processor(self, context):
        """
        Implements a state machine
        :return:
        """
        http_response = (yield)
        credssp_context = self._get_credssp_header(http_response)

        if credssp_context is None:
            raise Exception('The remote host did not respond with a \'www-authenticate\' header containing '
                            '\'CredSSP\' as an available authentication mechanism')

        # 1. First, secure the channel with a TLS Handshake
        if not credssp_context:
            self.tls_connection = SSL.Connection(self.tls_context)
            self.tls_connection.set_connect_state()
            while True:
                try:
                    self.tls_connection.do_handshake()
                except SSL.WantReadError:
                    http_response = yield self._set_credssp_header(http_response.request, self.tls_connection.bio_read(4096))
                    credssp_context = self._get_credssp_header(http_response)
                    if credssp_context is None or not credssp_context:
                        raise Exception('The remote host rejected the CredSSP TLS handshake')
                    self.tls_connection.bio_write(credssp_context)
                else:
                    break

        # add logging to display the negotiated cipher (move to a function)
        openssl_lib = _util.binding.lib
        ffi = _util.binding.ffi
        cipher = openssl_lib.SSL_get_current_cipher(self.tls_connection._ssl)
        cipher_name = ffi.string( openssl_lib.SSL_CIPHER_get_name(cipher))
        log.debug("Negotiated TLS Cipher: %s", cipher_name)

        # 2. Send an TSRequest containing an NTLM Negotiate Request
        context_generator = context.initialize_security_context()
        negotiate_token = context_generator.send(None)
        log.debug("NTLM Type 1: %s", AsHex(negotiate_token))

        ts_request = TSRequest()
        ts_request['negoTokens'] = negotiate_token
        self.tls_connection.send(ts_request.getData())

        http_response = yield self._set_credssp_header(http_response.request, self.tls_connection.bio_read(4096))

        # Extract and decrypt the encoded TSRequest response struct from the Negotiate header
        authenticate_header = self._get_credssp_header(http_response)
        if not authenticate_header or authenticate_header is None:
            raise Exception("The remote host rejected the CredSSP negotiation token")
        self.tls_connection.bio_write(authenticate_header)

        # NTLM Challenge Response and Server Public Key Validation
        ts_request = TSRequest()
        ts_request.fromString(self.tls_connection.recv(8192))
        challenge_token = ts_request['negoTokens']
        log.debug("NTLM Type 2: %s", AsHex(challenge_token))
        server_cert = self.tls_connection.get_peer_certificate()

        # not using channel bindings
        #certificate_digest = base64.b16decode(server_cert.digest('SHA256').replace(':', ''))
        ## channel_binding_structure = gss_channel_bindings_struct()
        ## channel_binding_structure['application_data'] = "tls-server-end-point:" + certificate_digest
        public_key = HttpCredSSPAuth._get_rsa_public_key(server_cert)
        # The _RSAPublicKey must be 'wrapped' using the negotiated GSSAPI mechanism and send to the server along with
        # the final SPNEGO token. This step of the CredSSP protocol is designed to thwart 'man-in-the-middle' attacks

        # Build and encrypt the response to the server
        ts_request = TSRequest()
        type3= context_generator.send(challenge_token)
        log.debug("NTLM Type 3: %s", AsHex(type3))
        ts_request['negoTokens'] = type3
        public_key_encrypted, signature = context.wrap_message(public_key)
        ts_request['pubKeyAuth'] = signature + public_key_encrypted

        self.tls_connection.send(ts_request.getData())
        enc_type3 = self.tls_connection.bio_read(8192)
        http_response = yield self._set_credssp_header(http_response.request, enc_type3)

        # TLS decrypt the response, then ASN decode and check the error code
        auth_response = self._get_credssp_header(http_response)
        if not auth_response or auth_response is None:
            raise Exception("The remote host rejected the challenge response")

        self.tls_connection.bio_write(auth_response)
        ts_request = TSRequest()
        ts_request.fromString(self.tls_connection.recv(8192))
        # TODO: determine how to validate server certificate here
        #a = ts_request['pubKeyAuth']
        # print ":".join("{:02x}".format(ord(c)) for c in a)

        # 4. Send the Credentials to be delegated, these are encrypted with both NTLM v2 and then by TLS
        tsp = TSPasswordCreds()
        tsp['domain'] = self.password_authenticator.get_domain()
        tsp['username'] = self.password_authenticator.get_username()
        tsp['password'] = self.password_authenticator.get_password()

        tsc = TSCredentials()
        tsc['type'] = 1
        tsc['credentials'] = tsp.getData()

        ts_request = TSRequest()
        encrypted, signature = context.wrap_message(tsc.getData())
        ts_request['authInfo'] = signature + encrypted

        self.tls_connection.send(ts_request.getData())
        token = self.tls_connection.bio_read(8192)

        http_response.request.body = self.body
        http_response = yield self._set_credssp_header(self._encrypt(http_response.request, self.tls_connection), token)

        if http_response.status_code == 401:
            raise Exception('Authentication Failed')