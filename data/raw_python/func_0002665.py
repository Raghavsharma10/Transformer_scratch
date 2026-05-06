def prepare_outdir(outdir):
    """
    Creates the output directory if not existing.
    If outdir is None or if no output_files are provided nothing happens.

    :param outdir: The output directory to create.
    """
    if outdir:
        outdir = os.path.expanduser(outdir)
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except os.error as e:
                raise JobExecutionError('Failed to create outdir "{}".\n{}'.format(outdir, str(e)))