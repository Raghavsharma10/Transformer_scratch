def map_single_end(credentials, instance_config, instance_name,
                   script_dir, index_dir, fastq_file, output_dir,
                   num_threads=None, seed_start_lmax=None,
                   mismatch_nmax=None, multimap_nmax=None,
                   splice_min_overhang=None,
                   out_mult_nmax=None, sort_bam=True, keep_unmapped=False,
                   self_destruct=True, compressed=True,
                   **kwargs):
    """Maps single-end reads using STAR.

    Reads are expected in FASTQ format. By default, they are also expected to
    be compressed with gzip.

    - recommended machine type: "n1-standard-16" (60 GB of RAM, 16 vCPUs).
    - recommended disk size: depends on size of FASTQ files, at least 128 GB.

    TODO: docstring"""

    if sort_bam:
        out_sam_type = 'BAM SortedByCoordinate'
    else:
        out_sam_type = 'BAM Unsorted'

    # template expects a list of FASTQ files
    fastq_files = fastq_file
    if isinstance(fastq_files, (str, _oldstr)):
        fastq_files = [fastq_file]

    template = _TEMPLATE_ENV.get_template(
        os.path.join('map_single-end.sh'))
    startup_script = template.render(
        script_dir=script_dir,
        index_dir=index_dir,
        fastq_files=fastq_files,
        output_dir=output_dir,
        num_threads=num_threads,
        seed_start_lmax=seed_start_lmax,
        self_destruct=self_destruct,
        mismatch_nmax=mismatch_nmax,
        multimap_nmax=multimap_nmax,
        splice_min_overhang=splice_min_overhang,
        out_mult_nmax=out_mult_nmax,
        keep_unmapped=keep_unmapped,
        compressed=compressed,
        out_sam_type=out_sam_type)

    if len(startup_script) > 32768:
        raise ValueError('Startup script larger than 32,768 bytes!')

    #print(startup_script)

    op_name = instance_config.create_instance(
        credentials, instance_name, startup_script=startup_script, **kwargs)

    return op_name