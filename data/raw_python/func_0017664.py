def move_to_output_dir(work_dir, output_dir, uuid=None, files=list()):
    """
    Moves files from work_dir to output_dir

    Input1: Working directory
    Input2: Output directory
    Input3: UUID to be preprended onto file name
    Input4: list of file names to be moved from working dir to output dir
    """
    for fname in files:
        if uuid is None:
            shutil.move(os.path.join(work_dir, fname), os.path.join(output_dir, fname))
        else:
            shutil.move(os.path.join(work_dir, fname), os.path.join(output_dir, '{}.{}'.format(uuid, fname)))