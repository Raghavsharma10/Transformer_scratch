def parse_sra(path_to_config):
    """
    Parses genetorrent config file.  Returns list of samples: [ [id1, id1 ], [id2, id2], ... ]
    Returns duplicate of ids to follow UUID/URL standard.
    """
    samples = []
    with open(path_to_config, 'r') as f:
        for line in f.readlines():
            if not line.isspace():
                samples.append(line.strip())
    return samples