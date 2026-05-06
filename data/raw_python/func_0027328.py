def split(compressed_data, segment_size, block_factory):
    """Splits compressed data into blocks.

    :param compressed_data: the compressed data to split
    :param segment_size: the size of a block in bytes
    :param block_factory: a BlockFactory used to construct the blocks

    :rtype: a list of block IDs of blocks that the block factory created while
    splitting
    """

    # Split compressed data into blocks
    segments = []

    current_segment_start = 0
    index = 0
    data_size = len(compressed_data)

    while index < data_size:
        current_byte = compressed_data[index]

        if index < data_size - 1:
            next_byte = compressed_data[index + 1]
        else:
            next_byte = None

        jump_size = 1

        if current_byte == RLE_BYTE:
            assert next_byte is not None, "Expected a command to follow " \
                "RLE byte"
            if next_byte == RLE_BYTE:
                jump_size = 2
            else:
                jump_size = 3

        elif current_byte == SPECIAL_BYTE:
            assert next_byte is not None, "Expected a command to follow " \
                "special byte"

            if next_byte == SPECIAL_BYTE:
                jump_size = 2
            elif next_byte == DEFAULT_INSTR_BYTE or \
                    next_byte == DEFAULT_WAVE_BYTE:
                jump_size = 3
            else:
                assert False, "Encountered unexpected EOF or block " \
                    "switch while segmenting"

        # Need two bytes for the jump or EOF
        if index - current_segment_start + jump_size > segment_size - 2:
            segments.append(compressed_data[
                current_segment_start:index])

            current_segment_start = index
        else:
            index += jump_size

    # Append the last segment, if any
    if current_segment_start != index:
        segments.append(compressed_data[
            current_segment_start:current_segment_start + index])

    # Make sure that no data was lost while segmenting
    total_segment_length = sum(map(len, segments))
    assert total_segment_length == len(compressed_data), "Lost %d bytes of " \
        "data while segmenting" % (len(compressed_data) - total_segment_length)

    block_ids = []

    for segment in segments:
        block = block_factory.new_block()
        block_ids.append(block.id)

    for (i, segment) in enumerate(segments):
        block = block_factory.blocks[block_ids[i]]

        assert len(block.data) == 0, "Encountered a block with "
        "pre-existing data while writing"

        if i == len(segments) - 1:
            # Write EOF to the end of the segment
            add_eof(segment)
        else:
            # Write a pointer to the next segment
            add_block_switch(segment, block_ids[i + 1])

        # Pad segment with zeroes until it's large enough
        pad(segment, segment_size)

        block.data = segment

    return block_ids