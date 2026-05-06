def parse_manifest(manifest_path):
    """
    Parse manifest file

    :param str manifest_path: Path to manifest file
    :return: samples
    :rtype: list[str, list]
    """
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if not line.isspace() and not line.startswith('#'):
                sample = line.strip().split('\t')
                require(2 <= len(sample) <= 3, 'Bad manifest format! '
                                               'Expected UUID\tURL1\t[URL2] (tab separated), got: {}'.format(sample))
                uuid = sample[0]
                urls = sample[1:]
                for url in urls:
                    require(urlparse(url).scheme and urlparse(url), 'Invalid URL passed for {}'.format(url))
                samples.append([uuid, urls])
    return samples