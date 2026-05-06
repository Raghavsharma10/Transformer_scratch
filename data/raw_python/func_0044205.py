def parsedata(self, packet):
        '''parse the data section of a packet, it can range from 0 to many bytes'''
        data = []
        datalength = ord(packet[3])
        position = 4
        while position < datalength + 4:
            data.append(packet[position])
            position += 1
        return data