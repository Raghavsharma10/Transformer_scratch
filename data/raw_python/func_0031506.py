def vcfutilsConsensus(outFile, vcfFile, id_, _, executor):
    """
    Use vcftools to extract consensus FASTA.

    @param outFile: The C{str} name to write the output to.
    @param vcfFile: The C{str} name of the VCF file with the calls from
        the pileup.
    @param id_: The C{str} identifier to use in the resulting FASTA sequence.
    @param executor: An C{Executor} instance.
    """
    executor.execute(
        'vcfutils.pl vcf2fq < %s | '
        'filter-fasta.py --fastq --quiet --saveAs fasta '
        '--idLambda \'lambda id: "%s"\' > %s' %
        (vcfFile, id_, outFile))