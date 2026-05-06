def request_issuance(self, csr):
        """
        Request a certificate.

        Authorizations should have already been completed for all of the names
        requested in the CSR.

        Note that unlike `acme.client.Client.request_issuance`, the certificate
        resource will have the body data as raw bytes.

        ..  seealso:: `txacme.util.csr_for_names`

        ..  todo:: Delayed issuance is not currently supported, the server must
                   issue the requested certificate immediately.

        :param csr: A certificate request message: normally
            `txacme.messages.CertificateRequest` or
            `acme.messages.CertificateRequest`.

        :rtype: Deferred[`acme.messages.CertificateResource`]
        :return: The issued certificate.
        """
        action = LOG_ACME_REQUEST_CERTIFICATE()
        with action.context():
            return (
                DeferredContext(
                    self._client.post(
                        self.directory[csr], csr,
                        content_type=DER_CONTENT_TYPE,
                        headers=Headers({b'Accept': [DER_CONTENT_TYPE]})))
                .addCallback(self._expect_response, http.CREATED)
                .addCallback(self._parse_certificate)
                .addActionFinish())