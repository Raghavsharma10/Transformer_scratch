def _createCert(self, hostname, serial):
        """
        Create a self-signed X.509 certificate.

        @type hostname: L{unicode}
        @param hostname: The hostname this certificate should be valid for.

        @type serial: L{int}
        @param serial: The serial number the certificate should have.

        @rtype: L{bytes}
        @return: The serialized certificate in PEM format.
        """
        privateKey = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend())
        publicKey = privateKey.public_key()
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .not_valid_before(datetime.today() - timedelta(days=1))
            .not_valid_after(datetime.today() + timedelta(days=365))
            .serial_number(serial)
            .public_key(publicKey)
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True)
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(hostname)]),
                critical=False)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False),
                critical=True)
            .add_extension(
                x509.ExtendedKeyUsage([
                    ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False)
            .sign(
                private_key=privateKey,
                algorithm=hashes.SHA256(),
                backend=default_backend()))
        return '\n'.join([
            privateKey.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()),
            certificate.public_bytes(
                encoding=serialization.Encoding.PEM),
            ])