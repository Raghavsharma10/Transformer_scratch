def variant_calling_and_qc(job, inputs, bam_id, bai_id):
    """
    Perform variant calling with samtools nad QC with CheckBias

    :param JobFunctionWrappingJob job: passed by Toil automatically
    :param Namespace inputs: Stores input arguments (see main)
    :param str bam_id: FileStore ID of bam
    :param str bai_id: FileStore ID of bam index file
    :return: FileStore ID of qc tarball
    :rtype: str
    """
    job.fileStore.logToMaster('Variant calling and QC: {}'.format(inputs.uuid))
    work_dir = job.fileStore.getLocalTempDir()
    # Pull in alignment.bam from fileStore
    job.fileStore.readGlobalFile(bam_id, os.path.join(work_dir, 'alignment.bam'))
    job.fileStore.readGlobalFile(bai_id, os.path.join(work_dir, 'alignment.bam.bai'))
    # Download input files
    input_info = [(inputs.genome, 'genome.fa'), (inputs.positions, 'positions.tsv'),
                  (inputs.genome_index, 'genome.fa.fai'), (inputs.gtf, 'annotation.gtf'),
                  (inputs.gtf_m53, 'annotation.m53')]
    for url, fname in input_info:
        download_url(job=job, url=url, work_dir=work_dir, name=fname)

    # Part 1: Variant Calling
    variant_command = ['mpileup',
                       '-f', 'genome.fa',
                       '-l', 'positions.tsv',
                       '-v', 'alignment.bam',
                       '-t', 'DP,SP,INFO/AD,INFO/ADF,INFO/ADR,INFO/DPR,SP',
                       '-o', '/data/output.vcf.gz']
    docker_call(job=job, work_dir=work_dir, parameters=variant_command,
                tool='quay.io/ucsc_cgl/samtools:1.3--256539928ea162949d8a65ca5c79a72ef557ce7c')

    # Part 2: QC
    qc_command = ['-o', 'qc',
                  '-n', 'alignment.bam',
                  '-a', 'annotation.gtf',
                  '-m', 'annotation.m53']
    docker_call(job=job, work_dir=work_dir, parameters=qc_command,
                tool='jvivian/checkbias:612f129--b08a1fb6526a620bbb0304b08356f2ae7c3c0ec3')
    # Write output to fileStore and return ids
    output_tsv = glob(os.path.join(work_dir, '*counts.tsv*'))[0]
    output_vcf = os.path.join(work_dir, 'output.vcf.gz')
    tarball_files('vcqc.tar.gz', file_paths=[output_tsv, output_vcf], output_dir=work_dir)
    return job.fileStore.writeGlobalFile(os.path.join(work_dir, 'vcqc.tar.gz'))