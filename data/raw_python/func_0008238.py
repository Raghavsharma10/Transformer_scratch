def init(main_dir: Path, logfile_path: Path, log_level: str):
    """
    Initialize the _downloader. TODO.

    :param main_dir: main directory
    :type main_dir: ~pathlib.Path
    :param logfile_path: logfile path
    :type logfile_path: ~pathlib.Path
    :param log_level: logging level
    :type log_level: str
    """
    dynamic_data.reset()
    dynamic_data.init_dirs(main_dir, logfile_path)

    dynamic_data.check_dirs()

    tools.create_dir_rec(dynamic_data.MAIN_DIR)
    tools.create_dir_rec(dynamic_data.TEMP_DIR)
    tools.create_dir_rec(dynamic_data.DOWNLOAD_DIR)
    tools.create_dir_rec(dynamic_data.SAVESTAT_DIR)
    tools.create_dir_rec(Path.resolve(dynamic_data.LOGFILE_PATH).parent)
    dynamic_data.LOG_LEVEL = log_level
    logging.basicConfig(filename=dynamic_data.LOGFILE_PATH, filemode='a', level=dynamic_data.LOG_LEVEL,
                        format='%(asctime)s.%(msecs)03d | %(levelname)s - %(name)s | %(module)s.%(funcName)s: %('
                               'message)s',
                        datefmt='%Y.%m.%d %H:%M:%S')
    logging.captureWarnings(True)

    cores = multiprocessing.cpu_count()
    dynamic_data.USING_CORES = min(4, max(1, cores - 1))

    info = f"{static_data.NAME} {static_data.VERSION}\n\n" \
           f"System: {platform.system()} - {platform.version()} - {platform.machine()} - {cores} cores\n" \
           f"Python: {platform.python_version()} - {' - '.join(platform.python_build())}\n" \
           f"Arguments: main={main_dir.resolve()} | logfile={logfile_path.resolve()} | loglevel={log_level}\n" \
           f"Using cores: {dynamic_data.USING_CORES}\n\n"
    with dynamic_data.LOGFILE_PATH.open(mode='w', encoding="utf8") as writer:
        writer.write(info)

    dynamic_data.AVAIL_PLUGINS = APlugin.get_plugins()