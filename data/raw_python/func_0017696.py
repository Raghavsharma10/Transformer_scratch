def consolidate_output(job, job_vars, output_ids):
    """
    Combine the contents of separate zipped outputs into one via streaming

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    output_ids: tuple   Nested tuple of all the output fileStore IDs
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    uuid = input_args['uuid']
    # Retrieve IDs
    rseq_id, exon_id, rsem_id = flatten(output_ids)
    # Retrieve output file paths to consolidate
    # map_tar = job.fileStore.readGlobalFile(map_id, os.path.join(work_dir, 'map.tar.gz'))
    qc_tar = job.fileStore.readGlobalFile(rseq_id, os.path.join(work_dir, 'qc.tar.gz'))
    exon_tar = job.fileStore.readGlobalFile(exon_id, os.path.join(work_dir, 'exon.tar.gz'))
    rsem_tar = job.fileStore.readGlobalFile(rsem_id, os.path.join(work_dir, 'rsem.tar.gz'))
    # I/O
    out_tar = os.path.join(work_dir, uuid + '.tar.gz')
    # Consolidate separate tarballs
    with tarfile.open(os.path.join(work_dir, out_tar), 'w:gz') as f_out:
        for tar in [rsem_tar, exon_tar, qc_tar]:
            with tarfile.open(tar, 'r') as f_in:
                for tarinfo in f_in:
                    with closing(f_in.extractfile(tarinfo)) as f_in_file:
                        if tar == qc_tar:
                            tarinfo.name = os.path.join(uuid, 'rseq_qc', os.path.basename(tarinfo.name))
                        else:
                            tarinfo.name = os.path.join(uuid, os.path.basename(tarinfo.name))
                        f_out.addfile(tarinfo, fileobj=f_in_file)
    # Move to output directory of selected
    if input_args['output_dir']:
        output_dir = input_args['output_dir']
        mkdir_p(output_dir)
        copy_to_output_dir(work_dir, output_dir, uuid=None, files=[uuid + '.tar.gz'])
    # Write output file to fileStore
    ids['uuid.tar.gz'] = job.fileStore.writeGlobalFile(out_tar)
    # If S3 bucket argument specified, upload to S3
    if input_args['s3_dir']:
        job.addChildJobFn(upload_output_to_s3, job_vars)