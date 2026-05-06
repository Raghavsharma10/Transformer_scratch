def process_bam(bam, **kwargs):
    """Combines metrics from bam after extraction.

    Processing function: calls pool of worker functions
    to extract from a bam file the following metrics:
    -lengths
    -aligned lengths
    -qualities
    -aligned qualities
    -mapping qualities
    -edit distances to the reference genome scaled by read length
    Returned in a pandas DataFrame
    """
    logging.info("Nanoget: Starting to collect statistics from bam file {}.".format(bam))
    samfile = check_bam(bam)
    chromosomes = samfile.references
    params = zip([bam] * len(chromosomes), chromosomes)
    with cfutures.ProcessPoolExecutor() as executor:
        datadf = pd.DataFrame(
            data=[res for sublist in executor.map(extract_from_bam, params) for res in sublist],
            columns=["readIDs", "quals", "aligned_quals", "lengths",
                     "aligned_lengths", "mapQ", "percentIdentity"]) \
            .dropna(axis='columns', how='all') \
            .dropna(axis='index', how='any')
    logging.info("Nanoget: bam {} contains {} primary alignments.".format(
        bam, datadf["lengths"].size))
    return ut.reduce_memory_usage(datadf)