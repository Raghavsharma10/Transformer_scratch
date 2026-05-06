def getContext(self):
        """Creates a context.

        This will make contexts using ``SSLv23_METHOD``. This is
        because OpenSSL thought it would be a good idea to have
        ``TLSv1_METHOD`` mean "only use TLSv1.0" -- specifically, it
        disables TLSv1.2. Since we don't want to use SSLv2 and v3, we
        set OP_NO_SSLv2|OP_NO_SSLv3. Additionally, we set
        OP_SINGLE_DH_USE.

        """
        ctx = Context(SSLv23_METHOD)
        ctx.use_certificate_file("cert.pem")
        ctx.use_privatekey_file("key.pem")
        ctx.load_tmp_dh("dhparam.pem")
        ctx.set_options(OP_SINGLE_DH_USE|OP_NO_SSLv2|OP_NO_SSLv3)
        ctx.set_verify(VERIFY_PEER, self._verify)
        return ctx