def calc_checksum(self, data):
        """ Calculate the checksum we need """

        output = 0
        length = len(data)
        for value in data:
            output += int(value, 16)

        output += length
        checksum = hex(output & int('0x007F', 16)).lstrip("0x")
        data.append(checksum)
        data.append('F7')
        return data