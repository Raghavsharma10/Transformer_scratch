def sequence(db, chrom, start, end):
    """
    return the sequence for a region using the UCSC DAS
    server. note the start is 1-based
    each feature will have it's own .sequence method which sends
    the correct start and end to this function.

    >>> sequence('hg18', 'chr2', 2223, 2230)
    'caacttag'
    """
    url = "http://genome.ucsc.edu/cgi-bin/das/%s" % db
    url += "/dna?segment=%s:%i,%i"
    xml = U.urlopen(url % (chrom, start, end)).read()
    return _seq_from_xml(xml)