def bcftoolsConsensus(outFile, vcfFile, id_, referenceFile, executor):
    """
    Use bcftools to extract consensus FASTA.

    @param outFile: The C{str} name to write the output to.
    @param vcfFile: The C{str} name of the VCF file with the calls from
        the pileup.
    @param id_: The C{str} identifier to use in the resulting FASTA sequence.
    @param referenceFile: The C{str} name of the FASTA file with the reference
        sequence.
    @param executor: An C{Executor} instance.
    """
    bgz = vcfFile + '.gz'
    executor.execute('bgzip -c %s > %s' % (vcfFile, bgz))
    executor.execute('tabix %s' % bgz)
    executor.execute(
        'bcftools consensus %s < %s | '
        'filter-fasta.py --idLambda \'lambda id: "%s"\' > %s' %
        (bgz, referenceFile, id_, outFile))