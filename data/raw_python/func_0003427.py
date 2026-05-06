def build_cert_chain(self, flags=SSL_BUILD_CHAIN_FLAG_NONE):
        u'''
        Used for server side only!

        :param flags:
        :return: 1 for success and 0 for failure
        '''
        retVal = SSL_CTX_build_cert_chain(self._ctx, flags)
        return retVal