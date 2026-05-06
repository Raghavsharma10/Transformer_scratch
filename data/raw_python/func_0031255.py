def _transform_chrom(chrom):
    """Helper function to obtain specific sort order."""
    try:
        c = int(chrom)
    except:
        if chrom in ['X', 'Y']:
            return chrom
        elif chrom == 'MT':
            return '_MT'  # sort to the end
        else:
            return '__' + chrom # sort to the very end
    else:
        # make sure numbered chromosomes are sorted numerically
        return '%02d' % c