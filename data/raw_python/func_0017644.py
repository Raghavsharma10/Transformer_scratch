def download_count_upload(job,
                          master_ip,
                          input_file,
                          output_file,
                          kmer_length,
                          spark_conf,
                          memory,
                          sudo):
    '''
    Runs k-mer counting.

    1. If the input file is located in S3, the file is copied into HDFS.
    2. If the input file is not in Parquet format, the file is converted into Parquet.
    3. The k-mers are counted and saved as text.
    4. If the output path is an S3 URL, the file is copied back to S3.

    :param job: Toil job
    :param input_file: URL/path to input file to count k-mers on
    :param output_file: URL/path to save k-mer counts at
    :param kmer_length: The length of k-mer substrings to count.
    :param spark_conf: Optional Spark configuration. If set, memory should \
    not be set.
    :param memory: Amount of memory to provided to Spark workers. Must be set \
    if spark_conf is not set.
    :param sudo: Whether or not to run Spark containers with sudo.

    :type job: toil.Job
    :type input_file: string
    :type output_file: string
    :type kmer_length: int or string
    :type spark_conf: list of string or None
    :type memory: int or None
    :type sudo: boolean
    '''

    if master_ip is not None:
        hdfs_dir = "hdfs://{0}:{1}/".format(master_ip, HDFS_MASTER_PORT)
    else:
        _log.warn('Master IP is not set. If default filesystem is not set, jobs may fail.')
        hdfs_dir = ""

    # if the file isn't already in hdfs, copy it in
    hdfs_input_file = hdfs_dir
    if input_file.startswith("s3://"):

        # append the s3 file name to our hdfs path
        hdfs_input_file += input_file.split("/")[-1]

        # run the download
        _log.info("Downloading input file %s to %s.", input_file, hdfs_input_file)
        call_conductor(job, master_ip, input_file, hdfs_input_file,
                       memory=memory, override_parameters=spark_conf)

    else:
        if not input_file.startswith("hdfs://"):
            _log.warn("If not in S3, input file (%s) expected to be in HDFS (%s).",
                      input_file, hdfs_dir)

    # where are we writing the output to? is it going to a location in hdfs or not?
    run_upload = True
    hdfs_output_file = hdfs_dir + "kmer_output.txt"
    if output_file.startswith(hdfs_dir):
        run_upload = False
        hdfs_output_file = output_file
    
    # do we need to convert to adam?
    if (hdfs_input_file.endswith('.bam') or
        hdfs_input_file.endswith('.sam') or
        hdfs_input_file.endswith('.fq') or
        hdfs_input_file.endswith('.fastq')):
        
        hdfs_tmp_file = hdfs_input_file

        # change the file extension to adam
        hdfs_input_file = '.'.join(hdfs_input_file.split('.')[:-1].append('adam'))

        # convert the file
        _log.info('Converting %s into ADAM format at %s.', hdfs_tmp_file, hdfs_input_file)
        call_adam(job, master_ip,
                  ['transform',
                   hdfs_tmp_file, hdfs_input_file],
                  memory=memory, override_parameters=spark_conf)
        
    # run k-mer counting
    _log.info('Counting %d-mers in %s, and saving to %s.',
              kmer_length, hdfs_input_file, hdfs_output_file)
    call_adam(job, master_ip,
              ['count_kmers',
               hdfs_input_file, hdfs_output_file,
               str(kmer_length)],
              memory=memory, override_parameters=spark_conf)

    # do we need to upload the file back? if so, run upload
    if run_upload:
        _log.info("Uploading output file %s to %s.", hdfs_output_file, output_file)
        call_conductor(job, master_ip, hdfs_output_file, output_file,
                       memory=memory, override_parameters=spark_conf)