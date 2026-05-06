def compress(raw_data):
    """Compress raw bytes with the filepack algorithm.

    :param raw_data: an array of raw data bytes to compress

    :rtype: a list of compressed bytes
    """
    raw_data = bytearray(raw_data)
    compressed_data = []

    data_size = len(raw_data)

    index = 0
    next_bytes = [-1, -1, -1]

    def is_default_instrument(index):
        if index + len(DEFAULT_INSTRUMENT_FILEPACK) > len(raw_data):
            return False

        instr_bytes = raw_data[index:index + len(DEFAULT_INSTRUMENT_FILEPACK)]

        if instr_bytes[0] != 0xa8 or instr_bytes[1] != 0:
            return False

        return instr_bytes == DEFAULT_INSTRUMENT_FILEPACK

    def is_default_wave(index):
        return (index + len(DEFAULT_WAVE) <= len(raw_data) and
                raw_data[index:index + len(DEFAULT_WAVE)] == DEFAULT_WAVE)

    while index < data_size:
        current_byte = raw_data[index]

        for i in range(3):
            if index < data_size - (i + 1):
                next_bytes[i] = raw_data[index + (i + 1)]
            else:
                next_bytes[i] = -1

        if current_byte == RLE_BYTE:
            compressed_data.append(RLE_BYTE)
            compressed_data.append(RLE_BYTE)
            index += 1
        elif current_byte == SPECIAL_BYTE:
            compressed_data.append(SPECIAL_BYTE)
            compressed_data.append(SPECIAL_BYTE)
            index += 1
        elif is_default_instrument(index):
            counter = 1
            index += len(DEFAULT_INSTRUMENT_FILEPACK)

            while (is_default_instrument(index) and
                   counter < 0x100):
                counter += 1
                index += len(DEFAULT_INSTRUMENT_FILEPACK)

            compressed_data.append(SPECIAL_BYTE)
            compressed_data.append(DEFAULT_INSTR_BYTE)
            compressed_data.append(counter)

        elif is_default_wave(index):
            counter = 1
            index += len(DEFAULT_WAVE)

            while is_default_wave(index) and counter < 0xff:
                counter += 1
                index += len(DEFAULT_WAVE)

            compressed_data.append(SPECIAL_BYTE)
            compressed_data.append(DEFAULT_WAVE_BYTE)
            compressed_data.append(counter)

        elif (current_byte == next_bytes[0] and
              next_bytes[0] == next_bytes[1] and
              next_bytes[1] == next_bytes[2]):
            # Do RLE compression

            compressed_data.append(RLE_BYTE)
            compressed_data.append(current_byte)
            counter = 0

            while (index < data_size and
                   raw_data[index] == current_byte and
                   counter < 0xff):
                index += 1
                counter += 1

            compressed_data.append(counter)
        else:
            compressed_data.append(current_byte)
            index += 1

    return compressed_data