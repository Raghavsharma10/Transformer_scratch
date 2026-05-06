def load_config():
    """Load configuration file containing API KEY and other settings.

    :rtype: str
    """

    configfile = get_configfile()

    if not os.path.exists(configfile):
        data = {
            'apikey': 'GET KEY AT: https://www.filemail.com/apidoc/ApiKey.aspx'
            }

        save_config(data)

    with open(configfile, 'rb') as f:
        return json.load(f)