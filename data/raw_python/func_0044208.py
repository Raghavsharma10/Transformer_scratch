def dispense(self):
        '''dispense a card if ready, otherwise throw an Exception'''
        self.sendcommand(Vendapin.DISPENSE)
        # wait for the reply
        time.sleep(1)
        # parse the reply
        response = self.receivepacket()
        print('Vendapin.dispense(): ' + str(response))
        if not self.was_packet_accepted(response):
            raise Exception('DISPENSE packet not accepted: ' + str(response))
        return self.parsedata(response)[0]