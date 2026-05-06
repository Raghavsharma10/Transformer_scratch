def send(self, alf):
    ''' Non-blocking send '''
    send_alf = SendThread(self.url, alf, self.connection_timeout, self.retry_count)
    send_alf.start()