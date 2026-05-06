def set_ecdh_curve(self, curve_name=None):
        u''' Select a curve to use for ECDH(E) key exchange or set it to auto mode

        Used for server only!

        s.a. openssl.exe ecparam -list_curves

        :param None | str curve_name: None = Auto-mode, "secp256k1", "secp384r1", ...
        :return: 1 for success and 0 for failure
        '''
        if curve_name:
            retVal = SSL_CTX_set_ecdh_auto(self._ctx, 0)
            avail_curves = get_elliptic_curves()
            key = [curve for curve in avail_curves if curve.name == curve_name][0].to_EC_KEY()
            retVal &= SSL_CTX_set_tmp_ecdh(self._ctx, key)
        else:
            retVal = SSL_CTX_set_ecdh_auto(self._ctx, 1)
        return retVal