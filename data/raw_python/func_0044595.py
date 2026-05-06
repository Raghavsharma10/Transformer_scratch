def run_naccess(pdb, mode, path=True, include_hetatms=False, outfile=None,
                path_to_ex=None):
    """Uses naccess to run surface accessibility calculations.

    Notes
    -----
    Requires the naccess program, with a path to its executable
    provided in global_settings. For information on the Naccess program,
    see: http://www.bioinf.manchester.ac.uk/naccess/
    This includes information on the licensing, which is not free for
    Industrial and Profit-making instituions.

    Parameters
    ----------
    pdb : str
        Path to pdb file or string.
    mode : str
        Return mode of naccess. One of 'asa', 'rsa' or 'log'.
    path : bool, optional
        Indicates if pdb is a path or a string.
    outfile : str, optional
        Filepath for storing the naccess output.
    path_to_ex : str or None
        Path to the binary for naccess, if none then it is assumed
        that the binary is available on the path as `naccess`.

    Returns
    -------
    naccess_out : str
        naccess output file for given mode as a string.
    """
    if mode not in ['asa', 'rsa', 'log']:
        raise ValueError(
            "mode {} not valid. Must be \'asa\', \'rsa\' or \'log\'"
            .format(mode))
    if path_to_ex:
        naccess_exe = path_to_ex
    else:
        naccess_exe = 'naccess'

    if not path:
        if type(pdb) == str:
            pdb = pdb.encode()
    else:
        with open(pdb, 'r') as inf:
            pdb = inf.read().encode()

    this_dir = os.getcwd()
    # temp pdb file in temp dir.
    temp_dir = tempfile.TemporaryDirectory()
    temp_pdb = tempfile.NamedTemporaryFile(dir=temp_dir.name)
    temp_pdb.write(pdb)
    temp_pdb.seek(0)
    # run naccess in the temp_dir. Files created will be written here.
    os.chdir(temp_dir.name)

    if include_hetatms:
        naccess_args = '-h'
        subprocess.check_output([naccess_exe, naccess_args, temp_pdb.name])
    else:
        subprocess.check_output([naccess_exe, temp_pdb.name])
    temp_pdb.close()
    with open('.{}'.format(mode), 'r') as inf:
        naccess_out = inf.read()
    # navigate back to initial directory and clean up.
    os.chdir(this_dir)
    if outfile:
        with open(outfile, 'w') as inf:
            inf.write(naccess_out)
    temp_dir.cleanup()

    return naccess_out