def bcftoolsMpileup(outFile, referenceFile, alignmentFile, executor):
    """
    Use bcftools mpileup to generate VCF.

    @param outFile: The C{str} name to write the output to.
    @param referenceFile: The C{str} name of the FASTA file with the reference
        sequence.
    @param alignmentFile: The C{str} name of the SAM or BAM alignment file.
    @param executor: An C{Executor} instance.
    """
    executor.execute(
        'bcftools mpileup -Ov -f %s %s > %s' %
        (referenceFile, alignmentFile, outFile))