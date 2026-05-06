def get_chromosome_lengths(fasta_file, fancy_sort=True):
    """Extract chromosome lengths from genome FASTA file."""
    chromlen = []
    with gzip.open(fasta_file, 'rt', encoding='ascii') as fh:
        fasta = SeqIO.parse(fh, 'fasta')
        for i, f in enumerate(fasta):
            chromlen.append((f.id, len(f.seq)))
            _LOGGER.info('Processed chromosome "%s"...', f.id)
            #print(dir(f))
            #if i == 1: break
            
    # convert to pandas Series
    chromlen = pd.Series(OrderedDict(chromlen))
    chromlen.index.name = 'Chromosome'
    chromlen.name = 'Length'

    if fancy_sort:
        # sort using fancy ordering
        chrom_for_sorting = chromlen.index.to_series().apply(_transform_chrom)
        a = chrom_for_sorting.argsort(kind='mergesort')
        chromlen = chromlen.iloc[a]

    return chromlen