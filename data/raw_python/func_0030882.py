def main(recordFilenames, fastaFilename, title, xRange, bitRange):
    """
    Print reads that match in a specified X-axis and bit score range.

    @param recordFilenames: A C{list} of C{str} file names contain results of a
        BLAST run, in JSON format.
    @param fastaFilename: The C{str} name of the FASTA file that was originally
        BLASTed.
    @param title: The C{str} title of the subject sequence, as output by BLAST.
    @param xRange: A (start, end) list of C{int}s, giving an X-axis range or
        C{None} if the entire X axis range should be printed.
    @param bitRange: A (start, end) list of C{int}s, giving a bit score range
        or C{None} if the entire bit score range should be printed.
    """
    reads = FastaReads(fastaFilename)
    blastReadsAlignments = BlastReadsAlignments(reads, recordFilenames)
    filtered = blastReadsAlignments.filter(whitelist=set([title]),
                                           negativeTitleRegex='.')
    titlesAlignments = TitlesAlignments(filtered)

    if title not in titlesAlignments:
        print('%s: Title %r not found in BLAST output' % (sys.argv[0], title))
        sys.exit(3)

    for titleAlignment in titlesAlignments[title]:
        for hsp in titleAlignment.hsps:
            if ((xRange is None or (xRange[0] <= hsp.subjectEnd and
                                    xRange[1] >= hsp.subjectStart)) and
                (bitRange is None or (bitRange[0] <= hsp.score.score <=
                                      bitRange[1]))):
                print(('query: %s, start: %d, end: %d, score: %d' % (
                       titleAlignment.read.id, hsp.subjectStart,
                       hsp.subjectEnd, hsp.score.score)))