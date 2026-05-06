def _config_bootstrap(self):
        """Go through and establish the defaults on the file system.

        The approach here was stolen from the CLI tool provided with the
        module. Idea being that the user should not always need to provide a
        username and password in order to run the script. If the configuration
        file is already present with valid data, then lets use it.
        """
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(CONFIG_PATH)
        if not os.path.exists(CONFIG_FILE):
            json.dump(CONFIG_DEFAULTS, open(CONFIG_FILE, 'w'), indent=4,
                      separators=(',', ': '))
        config = CONFIG_DEFAULTS
        if self._email and self._password:
            #  Save the configuration locally to pull later on
            config['email'] = self._email
            config['password'] = str(obfuscate(self._password, 'store'))
            self._log.debug("Caching authentication in config file")
            json.dump(config, open(CONFIG_FILE, 'w'), indent=4,
                      separators=(',', ': '))
        else:
            #  Load the config file and override the class
            config = json.load(open(CONFIG_FILE))
            if config.get('py2', PY2) != PY2:
                raise Exception("Python versions have changed. Please run `setup` again to reconfigure the client.")
            if config['email'] and config['password']:
                self._email = config['email']
                self._password = obfuscate(str(config['password']), 'fetch')
                self._log.debug("Loaded authentication from config file")