def set_curves(self, curves):
        u''' Set supported curves by name, nid or nist.

        :param str | tuple(int) curves: Example "secp384r1:secp256k1", (715, 714), "P-384", "K-409:B-409:K-571", ...
        :return: 1 for success and 0 for failure
        '''
        retVal = None
        if isinstance(curves, str):
            retVal = SSL_CTX_set1_curves_list(self._ctx, curves)
        elif isinstance(curves, tuple):
            retVal = SSL_CTX_set1_curves(self._ctx, curves, len(curves))
        return retVal