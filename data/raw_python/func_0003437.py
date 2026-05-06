def getpeercert(self, binary_form=False):
        """Retrieve the peer's certificate

        When binary form is requested, the peer's DER-encoded certficate is
        returned if it was transmitted during the handshake.

        When binary form is not requested, and the peer's certificate has been
        validated, then a certificate dictionary is returned. If the certificate
        was not validated, an empty dictionary is returned.

        In all cases, None is returned if no certificate was received from the
        peer.
        """

        try:
            peer_cert = _X509(SSL_get_peer_certificate(self._ssl.value))
        except openssl_error():
            return

        if binary_form:
            return i2d_X509(peer_cert.value)
        if self._cert_reqs == CERT_NONE:
            return {}
        return decode_cert(peer_cert)