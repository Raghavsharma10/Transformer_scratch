def load_lsdsng(filename):
    """Load a Project from a ``.lsdsng`` file.

    :param filename: the name of the file from which to load
    :rtype: :py:class:`pylsdj.Project`
    """

    # Load preamble data so that we know the name and version of the song
    with open(filename, 'rb') as fp:
        preamble_data = bread.parse(fp, spec.lsdsng_preamble)

    with open(filename, 'rb') as fp:
        # Skip the preamble this time around
        fp.seek(int(len(preamble_data) / 8))

        # Load compressed data into a block map and use BlockReader to
        # decompress it
        factory = BlockFactory()

        while True:
            block_data = bytearray(fp.read(blockutils.BLOCK_SIZE))

            if len(block_data) == 0:
                break

            block = factory.new_block()
            block.data = block_data

        remapped_blocks = filepack.renumber_block_keys(factory.blocks)

        reader = BlockReader()
        compressed_data = reader.read(remapped_blocks)

        # Now, decompress the raw data and use it and the preamble to construct
        # a Project
        raw_data = filepack.decompress(compressed_data)

        name = preamble_data.name
        version = preamble_data.version
        size_blks = int(math.ceil(
            float(len(compressed_data)) / blockutils.BLOCK_SIZE))

        return Project(name, version, size_blks, raw_data)