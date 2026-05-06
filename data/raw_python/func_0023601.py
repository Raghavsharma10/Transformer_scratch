def rdy(self, count):
        '''Indicate that you're ready to receive'''
        self.ready = count
        self.last_ready_sent = count
        return self.send(constants.RDY + ' ' + str(count))