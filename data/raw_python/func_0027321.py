def load_srm(filename):
    """Load a Project from an ``.srm`` file.

    :param filename: the name of the file from which to load
    :rtype: :py:class:`pylsdj.Project`
    """

    # .srm files are just decompressed projects without headers

    # In order to determine the file's size in compressed blocks, we have to
    # compress it first
    with open(filename, 'rb') as fp:
        raw_data = fp.read()

    compressed_data = filepack.compress(raw_data)

    factory = BlockFactory()
    writer = BlockWriter()
    writer.write(compressed_data, factory)

    size_in_blocks = len(factory.blocks)

    # We'll give the file a dummy name ("SRMLOAD") and version, since we know
    # neither
    name = "SRMLOAD"
    version = 0

    return Project(name, version, size_in_blocks, raw_data)