def consolidate_output(job, config, mutect, pindel, muse):
    """
    Combine the contents of separate tarball outputs into one via streaming

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Argparse Namespace object containing argument inputs
    :param str mutect: MuTect tarball FileStoreID
    :param str pindel: Pindel tarball FileStoreID
    :param str muse: MuSe tarball FileStoreID
    """
    work_dir = job.fileStore.getLocalTempDir()
    mutect_tar, pindel_tar, muse_tar = None, None, None
    if mutect:
        mutect_tar = job.fileStore.readGlobalFile(mutect, os.path.join(work_dir, 'mutect.tar.gz'))
    if pindel:
        pindel_tar = job.fileStore.readGlobalFile(pindel, os.path.join(work_dir, 'pindel.tar.gz'))
    if muse:
        muse_tar = job.fileStore.readGlobalFile(muse, os.path.join(work_dir, 'muse.tar.gz'))
    out_tar = os.path.join(work_dir, config.uuid + '.tar.gz')
    # Consolidate separate tarballs into one as streams (avoids unnecessary untaring)
    tar_list = [x for x in [mutect_tar, pindel_tar, muse_tar] if x is not None]
    with tarfile.open(os.path.join(work_dir, out_tar), 'w:gz') as f_out:
        for tar in tar_list:
            with tarfile.open(tar, 'r') as f_in:
                for tarinfo in f_in:
                    with closing(f_in.extractfile(tarinfo)) as f_in_file:
                        if tar is mutect_tar:
                            tarinfo.name = os.path.join(config.uuid, 'mutect', os.path.basename(tarinfo.name))
                        elif tar is pindel_tar:
                            tarinfo.name = os.path.join(config.uuid, 'pindel', os.path.basename(tarinfo.name))
                        else:
                            tarinfo.name = os.path.join(config.uuid, 'muse', os.path.basename(tarinfo.name))
                        f_out.addfile(tarinfo, fileobj=f_in_file)
    # Move to output location
    if urlparse(config.output_dir).scheme == 's3':
        job.fileStore.logToMaster('Uploading {} to S3: {}'.format(config.uuid, config.output_dir))
        s3am_upload(job=job, fpath=out_tar, s3_dir=config.output_dir, num_cores=config.cores)
    else:
        job.fileStore.logToMaster('Moving {} to output dir: {}'.format(config.uuid, config.output_dir))
        mkdir_p(config.output_dir)
        copy_files(file_paths=[out_tar], output_dir=config.output_dir)