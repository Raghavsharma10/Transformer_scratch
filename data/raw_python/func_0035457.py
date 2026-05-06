def is_success(self, check_timeout=True):
        '''
        Check if Webpay response ``TBK_RESPUESTA`` is equal to ``0`` and if the lapse between initialization
        and this call is less than ``self.timeout`` when ``check_timeout`` is ``True`` (default).

        :param check_timeout: When ``True``, check time between initialization and call.
        '''
        if check_timeout and self.is_timeout():
            return False
        return self.payload.response == self.payload.SUCCESS_RESPONSE_CODE