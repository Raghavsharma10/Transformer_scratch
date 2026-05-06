def gatk_haplotype_caller(job,
                          bam, bai,
                          ref, fai, ref_dict,
                          annotations=None,
                          emit_threshold=10.0, call_threshold=30.0,
                          unsafe_mode=False,
                          hc_output=None):
    """
    Uses GATK HaplotypeCaller to identify SNPs and INDELs. Outputs variants in a Genomic VCF file.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param str bam: FileStoreID for BAM file
    :param str bai: FileStoreID for BAM index file
    :param str ref: FileStoreID for reference genome fasta file
    :param str ref_dict: FileStoreID for reference sequence dictionary file
    :param str fai: FileStoreID for reference fasta index file
    :param list[str] annotations: List of GATK variant annotations, default is None
    :param float emit_threshold: Minimum phred-scale confidence threshold for a variant to be emitted, default is 10.0
    :param float call_threshold: Minimum phred-scale confidence threshold for a variant to be called, default is 30.0
    :param bool unsafe_mode: If True, runs gatk UNSAFE mode: "-U ALLOW_SEQ_DICT_INCOMPATIBILITY"
    :param str hc_output: URL or local path to pre-cooked VCF file, default is None
    :return: FileStoreID for GVCF file
    :rtype: str
    """
    job.fileStore.logToMaster('Running GATK HaplotypeCaller')

    inputs = {'genome.fa': ref,
              'genome.fa.fai': fai,
              'genome.dict': ref_dict,
              'input.bam': bam,
              'input.bam.bai': bai}

    work_dir = job.fileStore.getLocalTempDir()
    for name, file_store_id in inputs.iteritems():
        job.fileStore.readGlobalFile(file_store_id, os.path.join(work_dir, name))

    # Call GATK -- HaplotypeCaller with parameters to produce a genomic VCF file:
    # https://software.broadinstitute.org/gatk/documentation/article?id=2803
    command = ['-T', 'HaplotypeCaller',
               '-nct', str(job.cores),
               '-R', 'genome.fa',
               '-I', 'input.bam',
               '-o', 'output.g.vcf',
               '-stand_call_conf', str(call_threshold),
               '-stand_emit_conf', str(emit_threshold),
               '-variant_index_type', 'LINEAR',
               '-variant_index_parameter', '128000',
               '--genotyping_mode', 'Discovery',
               '--emitRefConfidence', 'GVCF']

    if unsafe_mode:
        command = ['-U', 'ALLOW_SEQ_DICT_INCOMPATIBILITY'] + command

    if annotations:
        for annotation in annotations:
            command.extend(['-A', annotation])

    # Uses docker_call mock mode to replace output with hc_output file
    outputs = {'output.g.vcf': hc_output}
    docker_call(job=job, work_dir=work_dir,
                env={'JAVA_OPTS': '-Djava.io.tmpdir=/data/ -Xmx{}'.format(job.memory)},
                parameters=command,
                tool='quay.io/ucsc_cgl/gatk:3.5--dba6dae49156168a909c43330350c6161dc7ecc2',
                inputs=inputs.keys(),
                outputs=outputs,
                mock=True if outputs['output.g.vcf'] else False)
    return job.fileStore.writeGlobalFile(os.path.join(work_dir, 'output.g.vcf'))