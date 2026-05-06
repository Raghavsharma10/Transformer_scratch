def star(job, inputs, r1_cutadapt, r2_cutadapt):
    """
    Performs alignment of fastqs to BAM via STAR

    :param JobFunctionWrappingJob job: passed by Toil automatically
    :param Namespace inputs: Stores input arguments (see main)
    :param str r1_cutadapt: FileStore ID of read 1 fastq
    :param str r2_cutadapt: FileStore ID of read 2 fastq
    """
    job.fileStore.logToMaster('Aligning with STAR: {}'.format(inputs.uuid))
    work_dir = job.fileStore.getLocalTempDir()
    cores = min(inputs.cores, 16)
    # Retrieve files
    job.fileStore.readGlobalFile(r1_cutadapt, os.path.join(work_dir, 'R1_cutadapt.fastq'))
    job.fileStore.readGlobalFile(r2_cutadapt, os.path.join(work_dir, 'R2_cutadapt.fastq'))
    # Get starIndex
    download_url(job=job, url=inputs.star_index, work_dir=work_dir, name='starIndex.tar.gz')
    subprocess.check_call(['tar', '-xvf', os.path.join(work_dir, 'starIndex.tar.gz'), '-C', work_dir])
    # Parameters
    parameters = ['--runThreadN', str(cores),
                  '--genomeDir', '/data/starIndex',
                  '--outFileNamePrefix', 'rna',
                  '--outSAMtype', 'BAM', 'SortedByCoordinate',
                  '--outSAMunmapped', 'Within',
                  '--quantMode', 'TranscriptomeSAM',
                  '--outSAMattributes', 'NH', 'HI', 'AS', 'NM', 'MD',
                  '--outFilterType', 'BySJout',
                  '--outFilterMultimapNmax', '20',
                  '--outFilterMismatchNmax', '999',
                  '--outFilterMismatchNoverReadLmax', '0.04',
                  '--alignIntronMin', '20',
                  '--alignIntronMax', '1000000',
                  '--alignMatesGapMax', '1000000',
                  '--alignSJoverhangMin', '8',
                  '--alignSJDBoverhangMin', '1',
                  '--sjdbScore', '1',
                  '--readFilesIn', '/data/R1_cutadapt.fastq', '/data/R2_cutadapt.fastq']
    # Call: STAR Map
    docker_call(job=job, tool='quay.io/ucsc_cgl/star:2.4.2a--bcbd5122b69ff6ac4ef61958e47bde94001cfe80',
                work_dir=work_dir, parameters=parameters)
    # Call Samtools Index
    index_command = ['index', '/data/rnaAligned.sortedByCoord.out.bam']
    docker_call(job=job, work_dir=work_dir, parameters=index_command,
                tool='quay.io/ucsc_cgl/samtools:1.3--256539928ea162949d8a65ca5c79a72ef557ce7c')
    # fileStore
    bam_id = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'rnaAligned.sortedByCoord.out.bam'))
    bai_id = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'rnaAligned.sortedByCoord.out.bam.bai'))
    job.fileStore.deleteGlobalFile(r1_cutadapt)
    job.fileStore.deleteGlobalFile(r2_cutadapt)
    # Launch children and follow-on
    vcqc_id = job.addChildJobFn(variant_calling_and_qc, inputs, bam_id, bai_id, cores=2, disk='30G').rv()
    spladder_id = job.addChildJobFn(spladder, inputs, bam_id, bai_id, disk='30G').rv()
    job.addFollowOnJobFn(consolidate_output_tarballs, inputs, vcqc_id, spladder_id, disk='30G')