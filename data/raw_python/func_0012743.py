def run(idf=None, weather=None, output_directory='', annual=False,
        design_day=False, idd=None, epmacro=False, expandobjects=False,
        readvars=False, output_prefix=None, output_suffix=None, version=False,
        verbose='v', ep_version=None):
    """
    Wrapper around the EnergyPlus command line interface.

    Parameters
    ----------
    idf : str
        Full or relative path to the IDF file to be run, or an IDF object.

    weather : str
        Full or relative path to the weather file.

    output_directory : str, optional
        Full or relative path to an output directory (default: 'run_outputs)

    annual : bool, optional
        If True then force annual simulation (default: False)

    design_day : bool, optional
        Force design-day-only simulation (default: False)

    idd : str, optional
        Input data dictionary (default: Energy+.idd in EnergyPlus directory)

    epmacro : str, optional
        Run EPMacro prior to simulation (default: False).

    expandobjects : bool, optional
        Run ExpandObjects prior to simulation (default: False)

    readvars : bool, optional
        Run ReadVarsESO after simulation (default: False)

    output_prefix : str, optional
        Prefix for output file names (default: eplus)

    output_suffix : str, optional
        Suffix style for output file names (default: L)
            L: Legacy (e.g., eplustbl.csv)
            C: Capital (e.g., eplusTable.csv)
            D: Dash (e.g., eplus-table.csv)

    version : bool, optional
        Display version information (default: False)

    verbose: str
        Set verbosity of runtime messages (default: v)
            v: verbose
            q: quiet

    ep_version: str
        EnergyPlus version, used to find install directory. Required if run() is
        called with an IDF file path rather than an IDF object.

    Returns
    -------
    str : status

    Raises
    ------
    CalledProcessError

    AttributeError
        If no ep_version parameter is passed when calling with an IDF file path
        rather than an IDF object.


    """
    args = locals().copy()
    # get unneeded params out of args ready to pass the rest to energyplus.exe
    verbose = args.pop('verbose')
    idf = args.pop('idf')
    iddname = args.get('idd')
    if not isinstance(iddname, str):
        args.pop('idd')
    try:
        idf_path = os.path.abspath(idf.idfname)
    except AttributeError:
        idf_path = os.path.abspath(idf)
    ep_version = args.pop('ep_version')
    # get version from IDF object or by parsing the IDF file for it
    if not ep_version:
        try:
            ep_version = '-'.join(str(x) for x in idf.idd_version[:3])
        except AttributeError:
            raise AttributeError(
                "The ep_version must be set when passing an IDF path. \
                Alternatively, use IDF.run()")

    eplus_exe_path, eplus_weather_path = install_paths(ep_version, iddname)
    if version:
        # just get EnergyPlus version number and return
        cmd = [eplus_exe_path, '--version']
        check_call(cmd)
        return

    # convert paths to absolute paths if required
    if os.path.isfile(args['weather']):
        args['weather'] = os.path.abspath(args['weather'])
    else:
        args['weather'] = os.path.join(eplus_weather_path, args['weather'])
    output_dir = os.path.abspath(args['output_directory'])
    args['output_directory'] = output_dir

    # store the directory we start in
    cwd = os.getcwd()
    run_dir = os.path.abspath(tempfile.mkdtemp())
    os.chdir(run_dir)

    # build a list of command line arguments
    cmd = [eplus_exe_path]
    for arg in args:
        if args[arg]:
            if isinstance(args[arg], bool):
                args[arg] = ''
            cmd.extend(['--{}'.format(arg.replace('_', '-'))])
            if args[arg] != "":
                cmd.extend([args[arg]])
    cmd.extend([idf_path])

    try:
        if verbose == 'v':
            print("\r\n" + " ".join(cmd) + "\r\n")
            check_call(cmd)
        elif verbose == 'q':
            check_call(cmd, stdout=open(os.devnull, 'w'))
    except CalledProcessError:
        message = parse_error(output_dir)
        raise EnergyPlusRunError(message)
    finally:
        os.chdir(cwd)
    return 'OK'