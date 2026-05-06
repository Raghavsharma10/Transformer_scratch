def parseColors(colors, args):
    """
    Parse read id color specification.

    @param colors: A C{list} of C{str}s. Each item is of the form, e.g.,
        'green X Y Z...', where each of X, Y, Z, ... etc. is either a read
        id or the name of a FASTA or FASTQ file containing reads whose ids
        should be displayed with the corresponding color. Note that if read
        ids contain spaces you will need to use the latter (i.e. FASTA/Q file
        name) approach because C{args.colors} is split on whitespace.
    @param args: The argparse C{Namespace} instance holding the other parsed
        command line arguments.
    @return: A C{dict} whose keys are colors and whose values are sets of
        read ids.
    """
    result = defaultdict(set)
    for colorInfo in colors:
        readIds = colorInfo.split()
        color = readIds.pop(0)
        for readId in readIds:
            if os.path.isfile(readId):
                filename = readId
                if args.fasta:
                    reads = FastaReads(filename)
                else:
                    reads = FastqReads(filename)
                for read in reads:
                    result[color].add(read.id)
            else:
                result[color].add(readId)
    return result