def delete(self, instance):
        '''Delete an instance'''
        flushdb(self.client) if flushdb else self.client.flushdb()