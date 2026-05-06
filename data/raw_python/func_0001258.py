def setup_logging(filename, log_dir=None, force_setup=False):
    ''' Try to load logging configuration from a file. Set level to INFO if failed.
    '''
    if not force_setup and ChirpCLI.SETUP_COMPLETED:
        logging.debug("Master logging has been setup. This call will be ignored.")
        return
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.isfile(filename):
        with open(filename) as config_file:
            try:
                config = json.load(config_file)
                logging.config.dictConfig(config)
                logging.info("logging was setup using {}".format(filename))
                ChirpCLI.SETUP_COMPLETED = True
            except Exception as e:
                logging.exception("Could not load logging config")
                # default logging config
                logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)