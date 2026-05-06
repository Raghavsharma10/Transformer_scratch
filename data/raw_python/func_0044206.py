def sendcommand(self, command, datalength=0, data=None):
        '''send a packet in the vendapin format'''
        packet = chr(Vendapin.STX) + chr(Vendapin.ADD) + chr(command) + chr(datalength)
        if datalength > 0:
            packet += chr(data)
        packet += chr(Vendapin.ETX)
        sendpacket = packet + chr(self._checksum(packet))
        self._printpacket(sendpacket)
        self.serial.write(sendpacket)