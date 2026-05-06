def request_status(self):
        '''request the status of the card dispenser and return the status code'''
        self.sendcommand(Vendapin.REQUEST_STATUS)
        # wait for the reply
        time.sleep(1)
        response = self.receivepacket()
        if self.was_packet_accepted(response):
            return Vendapin.READY
        else:
            return self.parsedata(response)[0]