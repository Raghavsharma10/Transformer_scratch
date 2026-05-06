def set_ssl_logging(self, enable=False, func=_ssl_logging_cb):
        u''' Enable or disable SSL logging

        :param True | False enable: Enable or disable SSL logging
        :param func: Callback function for logging
        '''
        if enable:
            SSL_CTX_set_info_callback(self._ctx, func)
        else:
            SSL_CTX_set_info_callback(self._ctx, 0)