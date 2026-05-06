def _line_to_entry(self,line):
    """parse the line into entries and keys"""
    f = line.rstrip().split("\t")
    """
    'chrom'
    'chromStart'
    'chromEnd'
    'name'
    'score'
    'strand'
    'thickStart'
    'thickEnd'
    'itemRgb'
    'blockCount'
    'blockSizes'
    'blockStarts'
    """
    return Bed12Fields(
       f[0],
       int(f[1]),
       int(f[2]),
       f[3],
       int(f[4]),
       f[5],
       int(f[6]),
       int(f[7]),
       [int(x) for x in f[8].rstrip(',').split(',')],
       int(f[9]),
       [int(x) for x in f[10].rstrip(',').split(',')],
       [int(x) for x in f[11].rstrip(',').split(',')])