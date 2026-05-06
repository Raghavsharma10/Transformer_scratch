def merge(blocks):
    """Merge the given blocks into a contiguous block of compressed data.

    :param blocks: the list of blocks
    :rtype: a list of compressed bytes
    """

    current_block = blocks[sorted(blocks.keys())[0]]

    compressed_data = []
    eof = False

    while not eof:
        data_size_to_append = None
        next_block = None

        i = 0
        while i < len(current_block.data) - 1:
            current_byte = current_block.data[i]
            next_byte = current_block.data[i + 1]

            if current_byte == RLE_BYTE:
                if next_byte == RLE_BYTE:
                    i += 2
                else:
                    i += 3
            elif current_byte == SPECIAL_BYTE:
                if next_byte in SPECIAL_DEFAULTS:
                    i += 3
                elif next_byte == SPECIAL_BYTE:
                    i += 2
                else:
                    data_size_to_append = i

                    # hit end of file
                    if next_byte == EOF_BYTE:
                        eof = True
                    else:
                        next_block = blocks[next_byte]

                    break
            else:
                i += 1

        assert data_size_to_append is not None, "Ran off the end of a "\
            "block without encountering a block switch or EOF"

        compressed_data.extend(current_block.data[0:data_size_to_append])

        if not eof:
            assert next_block is not None, "Switched blocks, but did " \
                "not provide the next block to switch to"

            current_block = next_block

    return compressed_data