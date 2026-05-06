def _get_rsa_public_key(cert):
        """
        PyOpenSSL does not provide a public method to export the public key from a certificate as a properly formatted
        ASN.1 RSAPublicKey structure. There are 'hacks' which use dump_privatekey(crypto.FILETYPE_ASN1, <public_key>),
        but this dumps the public key within a PrivateKeyInfo structure which is not suitable for a comparison. This
        approach uses the PyOpenSSL CFFI bindings to invoke the i2d_RSAPublicKey() which correctly extracts the key
        material in an ASN.1 RSAPublicKey structure.
        :param cert: The ASN.1 Encoded Certificate
        :return: The ASN.1 Encoded RSAPublicKey structure containing the supplied certificates public Key
        """
        openssl_pkey = cert.get_pubkey()
        openssl_lib = _util.binding.lib
        ffi = _util.binding.ffi
        buf = ffi.new("unsigned char **")
        rsa = openssl_lib.EVP_PKEY_get1_RSA(openssl_pkey._pkey)
        length = openssl_lib.i2d_RSAPublicKey(rsa, buf)
        public_key = ffi.buffer(buf[0], length)[:]
        ffi.gc(buf[0], openssl_lib.OPENSSL_free)
        return public_key