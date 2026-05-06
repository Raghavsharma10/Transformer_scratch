def req(self, message_id, timeout):
        '''Re-queue a message'''
        return self.send(constants.REQ + ' ' + message_id + ' ' + str(timeout))