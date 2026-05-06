def init_interface():
    sys.stdout = LoggerWriter(LOGGER.debug)
    sys.stderr = LoggerWriter(LOGGER.error)

    """
    Grab the ~/.polyglot/.env file for variables
    If you are running Polyglot v2 on this same machine
    then it should already exist. If not create it.
    """
    warnings.simplefilter('error', UserWarning)
    try:
        load_dotenv(join(expanduser("~") + '/.polyglot/.env'))
    except (UserWarning) as err:
        LOGGER.warning('File does not exist: {}.'.format(join(expanduser("~") + '/.polyglot/.env')), exc_info=True)
        # sys.exit(1)
    warnings.resetwarnings()

    """
    If this NodeServer is co-resident with Polyglot it will receive a STDIN config on startup
    that looks like:
    {"token":"2cb40e507253fc8f4cbbe247089b28db79d859cbed700ec151",
    "mqttHost":"localhost","mqttPort":"1883","profileNum":"10"}
    """

    init = select.select([sys.stdin], [], [], 1)[0]
    if init:
        line = sys.stdin.readline()
        try:
            line = json.loads(line)
            os.environ['PROFILE_NUM'] = line['profileNum']
            os.environ['MQTT_HOST'] = line['mqttHost']
            os.environ['MQTT_PORT'] = line['mqttPort']
            os.environ['TOKEN'] = line['token']
            LOGGER.info('Received Config from STDIN.')
        except (Exception) as err:
            # e = sys.exc_info()[0]
            LOGGER.error('Invalid formatted input. Skipping. %s', err, exc_info=True)