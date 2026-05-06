def event_payment(self, date, time, pid, commerce_id, transaction_id, request_ip, token, webpay_server):
        '''Record the payment event

        Official handler writes this information to TBK_EVN%Y%m%d file.
        '''
        raise NotImplementedError("Logging Handler must implement event_payment")