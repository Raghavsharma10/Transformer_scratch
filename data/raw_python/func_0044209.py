def reset(self, hard=False):
        '''reset the card dispense, either soft or hard based on boolean 2nd arg'''
        if hard:
            self.sendcommand(Vendapin.RESET, 1, 0x01)
            time.sleep(2)
        else:
            self.sendcommand(Vendapin.RESET)
            time.sleep(2)
            # parse the reply
            response = self.receivepacket()
            print('Vendapin.reset(soft): ' + str(response))