def splitFASTA(params):
    """
    Read the FASTA file named params['fastaFile'] and print out its
    sequences into files named 0.fasta, 1.fasta, etc. with
    params['seqsPerJob'] sequences per file.
    """
    assert params['fastaFile'][-1] == 'a', ('You must specify a file in '
                                            'fasta-format that ends in '
                                            '.fasta')

    fileCount = count = seqCount = 0
    outfp = None
    with open(params['fastaFile']) as infp:
        for seq in SeqIO.parse(infp, 'fasta'):
            seqCount += 1
            if count == params['seqsPerJob']:
                outfp.close()
                count = 0
            if count == 0:
                outfp = open('%d.fasta' % fileCount, 'w')
                fileCount += 1
            count += 1
            outfp.write('>%s\n%s\n' % (seq.description, str(seq.seq)))
    outfp.close()
    return fileCount, seqCount