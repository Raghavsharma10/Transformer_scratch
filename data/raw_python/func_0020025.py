def crc8(data):
        """
        Perform the 1-Wire CRC check on the provided data.

        :param bytearray data: 8 byte array representing 64 bit ROM code
        """
        crc = 0

        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x01:
                    crc = (crc >> 1) ^ 0x8C
                else:
                    crc >>= 1
                crc &= 0xFF
        return crc