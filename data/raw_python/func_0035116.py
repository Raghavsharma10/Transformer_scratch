def event_confirmation(self, date, time, pid, commerce_id, transaction_id, request_ip, order_id):
        '''Record the confirmation event.

        Official handler writes this information to TBK_EVN%Y%m%d file.
        '''
        raise NotImplementedError("Logging Handler must implement event_confirmation")