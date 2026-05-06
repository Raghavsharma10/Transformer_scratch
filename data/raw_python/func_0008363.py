def init_dirs(main_dir: Path, logfilepath: Path):
    """
    Initialize the main directories.

    :param main_dir: main directory
    :type main_dir: ~pathlib.Path
    :param logfilepath: log file
    :type logfilepath: ~pathlib.Path
    """
    global MAIN_DIR, TEMP_DIR, DOWNLOAD_DIR, SAVESTAT_DIR, LOGFILE_PATH
    MAIN_DIR = main_dir
    TEMP_DIR = MAIN_DIR.joinpath(Path('temp/'))
    DOWNLOAD_DIR = MAIN_DIR.joinpath(Path('downloads/'))
    SAVESTAT_DIR = MAIN_DIR.joinpath(Path('savestates/'))
    LOGFILE_PATH = MAIN_DIR.joinpath(logfilepath)