def fetch_chain(self, certr, max_length=10):
        """
        Fetch the intermediary chain for a certificate.

        :param acme.messages.CertificateResource certr: The certificate to
            fetch the chain for.
        :param int max_length: The maximum length of the chain that will be
            fetched.

        :rtype: Deferred[List[`acme.messages.CertificateResource`]]
        :return: The issuer certificate chain, ordered with the trust anchor
                 last.
        """
        action = LOG_ACME_FETCH_CHAIN()
        with action.context():
            if certr.cert_chain_uri is None:
                return succeed([])
            elif max_length < 1:
                raise errors.ClientError('chain too long')
            return (
                DeferredContext(
                    self._client.get(
                        certr.cert_chain_uri,
                        content_type=DER_CONTENT_TYPE,
                        headers=Headers({b'Accept': [DER_CONTENT_TYPE]})))
                .addCallback(self._parse_certificate)
                .addCallback(
                    lambda issuer:
                    self.fetch_chain(issuer, max_length=max_length - 1)
                    .addCallback(lambda chain: [issuer] + chain))
                .addActionFinish())