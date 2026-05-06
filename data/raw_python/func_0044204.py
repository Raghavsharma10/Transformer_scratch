def was_packet_accepted(self, packet):
        '''parse the "command" byte from the response packet to get a "response code"'''
        self._validatepacket(packet)
        cmd = ord(packet[2])
        if cmd == Vendapin.ACK: # Accepted/Positive Status
            return True
        elif cmd == Vendapin.NAK: # Rejected/Negative Status
            print('NAK - Rejected/Negative Status')
            return False
        elif cmd == Vendapin.INC: # Incomplete Command Packet
            raise Exception('INC - Incomplete Command Packet')
        elif cmd == Vendapin.UNR: # Unrecognized Command Packet
            raise Exception('UNR - Unrecognized Command Packet')
        elif cmd == Vendapin.CER: # Data Packet Checksum Error
            raise Exception('CER - Data Packet Checksum Error')
        else:
            raise Exception('Received bad CMD in response from card dispenser')