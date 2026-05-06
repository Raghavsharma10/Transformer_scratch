def decompress(compressed_data):
    """Decompress data that has been compressed by the filepack algorithm.

    :param compressed_data: an array of compressed data bytes to decompress

    :rtype: an array of decompressed bytes"""
    raw_data = []

    index = 0

    while index < len(compressed_data):
        current = compressed_data[index]
        index += 1

        if current == RLE_BYTE:
            directive = compressed_data[index]
            index += 1

            if directive == RLE_BYTE:
                raw_data.append(RLE_BYTE)
            else:
                count = compressed_data[index]
                index += 1

                raw_data.extend([directive] * count)
        elif current == SPECIAL_BYTE:
            directive = compressed_data[index]
            index += 1

            if directive == SPECIAL_BYTE:
                raw_data.append(SPECIAL_BYTE)
            elif directive == DEFAULT_WAVE_BYTE:
                count = compressed_data[index]
                index += 1

                raw_data.extend(DEFAULT_WAVE * count)
            elif directive == DEFAULT_INSTR_BYTE:
                count = compressed_data[index]
                index += 1

                raw_data.extend(DEFAULT_INSTRUMENT_FILEPACK * count)
            elif directive == EOF_BYTE:
                assert False, ("Unexpected EOF command encountered while "
                               "decompressing")
            else:
                assert False, "Countered unexpected sequence 0x%02x 0x%02x" % (
                    current, directive)
        else:
            raw_data.append(current)

    return raw_data