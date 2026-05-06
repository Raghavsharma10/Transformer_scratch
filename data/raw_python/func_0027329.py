def renumber_block_keys(blocks):
    """Renumber a block map's indices so that tehy match the blocks' block
    switch statements.

    :param blocks a block map to renumber
    :rtype: a renumbered copy of the block map
    """

    # There is an implicit block switch to the 0th block at the start of the
    # file
    byte_switch_keys = [0]
    block_keys = list(blocks.keys())

    # Scan the blocks, recording every block switch statement
    for block in list(blocks.values()):
        i = 0
        while i < len(block.data) - 1:
            current_byte = block.data[i]
            next_byte = block.data[i + 1]

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
                    if next_byte != EOF_BYTE:
                        byte_switch_keys.append(next_byte)

                    break

            else:
                i += 1

    byte_switch_keys.sort()
    block_keys.sort()

    assert len(byte_switch_keys) == len(block_keys), (
        "Number of blocks that are target of block switches (%d) "
        % (len(byte_switch_keys)) +
        "does not equal number of blocks in the song (%d)"
        % (len(block_keys)) +
        "; possible corruption")

    if byte_switch_keys == block_keys:
        # No remapping necessary
        return blocks

    new_block_map = {}

    for block_key, byte_switch_key in zip(
            block_keys, byte_switch_keys):

        new_block_map[byte_switch_key] = blocks[block_key]

    return new_block_map