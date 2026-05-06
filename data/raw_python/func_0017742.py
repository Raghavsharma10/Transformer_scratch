def download_reference_files(job, inputs, samples):
    """
    Downloads shared files that are used by all samples for alignment, or generates them if they were not provided.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace inputs: Input arguments (see main)
    :param list[list[str, list[str, str]]] samples: Samples in the format [UUID, [URL1, URL2]]
    """
    # Create dictionary to store FileStoreIDs of shared input files
    shared_ids = {}
    urls = [('amb', inputs.amb), ('ann', inputs.ann), ('bwt', inputs.bwt),
            ('pac', inputs.pac), ('sa', inputs.sa)]
    # Alt file is optional and can only be provided, not generated
    if inputs.alt:
        urls.append(('alt', inputs.alt))
    # Download reference
    download_ref = job.wrapJobFn(download_url_job, inputs.ref, disk='3G')  # Human genomes are typically ~3G
    job.addChild(download_ref)
    shared_ids['ref'] = download_ref.rv()
    # If FAI is provided, download it. Otherwise, generate it
    if inputs.fai:
        shared_ids['fai'] = job.addChildJobFn(download_url_job, inputs.fai).rv()
    else:
        faidx = job.wrapJobFn(run_samtools_faidx, download_ref.rv())
        shared_ids['fai'] = download_ref.addChild(faidx).rv()
    # If all BWA index files are provided, download them. Otherwise, generate them
    if all(x[1] for x in urls):
        for name, url in urls:
            shared_ids[name] = job.addChildJobFn(download_url_job, url).rv()
    else:
        job.fileStore.logToMaster('BWA index files not provided, creating now')
        bwa_index = job.wrapJobFn(run_bwa_index, download_ref.rv())
        download_ref.addChild(bwa_index)
        for x, name in enumerate(['amb', 'ann', 'bwt', 'pac', 'sa']):
            shared_ids[name] = bwa_index.rv(x)

    # Map_job distributes one sample in samples to the downlaod_sample_and_align function
    job.addFollowOnJobFn(map_job, download_sample_and_align, samples, inputs, shared_ids)