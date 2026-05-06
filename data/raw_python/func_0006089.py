def get_firefox_binary():
    """Gets the firefox binary

    @rtype: FirefoxBinary
    """
    browser_config = BrowserConfig()
    constants_config = ConstantsConfig()
    log_dir = os.path.join(constants_config.get('logs_dir'), 'firefox')
    create_directory(log_dir)

    log_path = os.path.join(log_dir, '{}_{}.log'.format(datetime.datetime.now().isoformat('_'), words.random_word()))
    log_file = open(log_path, 'w')
    log('Firefox log file: {}'.format(log_path))

    binary = FirefoxBinary(log_file=log_file)

    return binary