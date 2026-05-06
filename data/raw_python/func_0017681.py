def program_checks(job, input_args):
    """
    Checks that dependency programs are installed.

    input_args: dict        Dictionary of input arguments (from main())
    """
    # Program checks
    for program in ['curl', 'docker', 'unzip', 'samtools']:
        assert which(program), 'Program "{}" must be installed on every node.'.format(program)
    job.addChildJobFn(download_shared_files, input_args)