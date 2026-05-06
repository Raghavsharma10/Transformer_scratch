def consolidate_output_tarballs(job, inputs, vcqc_id, spladder_id):
    """
    Combine the contents of separate tarballs into one.

    :param JobFunctionWrappingJob job: passed by Toil automatically
    :param Namespace inputs: Stores input arguments (see main)
    :param str vcqc_id: FileStore ID of variant calling and QC tarball
    :param str spladder_id: FileStore ID of spladder tarball
    """
    job.fileStore.logToMaster('Consolidating files and uploading: {}'.format(inputs.uuid))
    work_dir = job.fileStore.getLocalTempDir()
    # Retrieve IDs
    uuid = inputs.uuid
    # Unpack IDs
    # Retrieve output file paths to consolidate
    vcqc_tar = job.fileStore.readGlobalFile(vcqc_id, os.path.join(work_dir, 'vcqc.tar.gz'))
    spladder_tar = job.fileStore.readGlobalFile(spladder_id, os.path.join(work_dir, 'spladder.tar.gz'))
    # I/O
    fname = uuid + '.tar.gz' if not inputs.improper_pair else 'IMPROPER_PAIR' + uuid + '.tar.gz'
    out_tar = os.path.join(work_dir, fname)
    # Consolidate separate tarballs into one
    with tarfile.open(os.path.join(work_dir, out_tar), 'w:gz') as f_out:
        for tar in [vcqc_tar, spladder_tar]:
            with tarfile.open(tar, 'r') as f_in:
                for tarinfo in f_in:
                    with closing(f_in.extractfile(tarinfo)) as f_in_file:
                        if tar == vcqc_tar:
                            tarinfo.name = os.path.join(uuid, 'variants_and_qc', os.path.basename(tarinfo.name))
                        else:
                            tarinfo.name = os.path.join(uuid, 'spladder', os.path.basename(tarinfo.name))
                        f_out.addfile(tarinfo, fileobj=f_in_file)
    # Move to output directory
    if inputs.output_dir:
        mkdir_p(inputs.output_dir)
        shutil.copy(out_tar, os.path.join(inputs.output_dir, os.path.basename(out_tar)))
    # Upload to S3
    if inputs.output_s3_dir:
        out_id = job.fileStore.writeGlobalFile(out_tar)
        job.addChildJobFn(s3am_upload_job, file_id=out_id, s3_dir=inputs.output_s3_dir,
                          file_name=fname, key_path=inputs.ssec, cores=inputs.cores)