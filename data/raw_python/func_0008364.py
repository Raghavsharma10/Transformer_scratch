def reset():
    """
    Reset all dynamic variables to the default values.
    """
    global MAIN_DIR, TEMP_DIR, DOWNLOAD_DIR, SAVESTAT_DIR, LOGFILE_PATH, USING_CORES, LOG_LEVEL, DISABLE_TQDM, \
        SAVE_STATE_VERSION
    MAIN_DIR = Path('./')
    TEMP_DIR = MAIN_DIR.joinpath(Path('temp/'))
    DOWNLOAD_DIR = MAIN_DIR.joinpath(Path('downloads/'))
    SAVESTAT_DIR = MAIN_DIR.joinpath(Path('savestates/'))
    LOGFILE_PATH = MAIN_DIR.joinpath(Path('UniDown.log'))

    USING_CORES = 1
    LOG_LEVEL = 'INFO'
    DISABLE_TQDM = False

    SAVE_STATE_VERSION = Version('1')