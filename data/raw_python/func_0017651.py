def parse_manifest(path_to_manifest):
    """
    Parses manifest file for Toil Germline Pipeline

    :param str path_to_manifest: Path to sample manifest file
    :return: List of GermlineSample namedtuples
    :rtype: list[GermlineSample]
    """
    bam_re = r"^(?P<uuid>\S+)\s(?P<url>\S+[bsc][r]?am)"
    fq_re = r"^(?P<uuid>\S+)\s(?P<url>\S+)\s(?P<paired_url>\S+)?\s?(?P<rg_line>@RG\S+)"
    samples = []
    with open(path_to_manifest, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            bam_match = re.match(bam_re, line)
            fastq_match = re.match(fq_re, line)
            if bam_match:
                uuid = bam_match.group('uuid')
                url = bam_match.group('url')
                paired_url = None
                rg_line = None
                require('.bam' in url.lower(),
                        'Expected .bam extension:\n{}:\t{}'.format(uuid, url))
            elif fastq_match:
                uuid = fastq_match.group('uuid')
                url = fastq_match.group('url')
                paired_url = fastq_match.group('paired_url')
                rg_line = fastq_match.group('rg_line')
                require('.fq' in url.lower() or '.fastq' in url.lower(),
                        'Expected .fq extension:\n{}:\t{}'.format(uuid, url))
            else:
                raise ValueError('Could not parse entry in manifest: %s\n%s' % (f.name, line))
            # Checks that URL has a scheme
            require(urlparse(url).scheme, 'Invalid URL passed for {}'.format(url))
            samples.append(GermlineSample(uuid, url, paired_url, rg_line))
    return samples