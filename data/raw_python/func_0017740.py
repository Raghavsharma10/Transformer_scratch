def parse_manifest(path_to_manifest):
    """
    Parses samples, specified in either a manifest or listed with --samples

    :param str path_to_manifest: Path to configuration file
    :return: Samples and their attributes as defined in the manifest
    :rtype: list[list]
    """
    samples = []
    with open(path_to_manifest, 'r') as f:
        for line in f.readlines():
            if not line.isspace() and not line.startswith('#'):
                sample = line.strip().split('\t')
                require(len(sample) == 3, 'Bad manifest format! '
                                          'Expected 3 tab separated columns, got: {}'.format(sample))
                uuid, normal, tumor = sample
                for url in [normal, tumor]:
                    require(urlparse(url).scheme and urlparse(url), 'Invalid URL passed for {}'.format(url))
                samples.append(sample)
    return samples