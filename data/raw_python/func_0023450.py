def req(self, timeout):
        '''Re-queue a message'''
        self.connection.req(self.id, timeout)
        self.processed = True