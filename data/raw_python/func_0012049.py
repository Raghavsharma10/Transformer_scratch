def load(self, file=CONFIG_FILE):
        """
        load a configuration file. loads default config if file is not found
        """
        if not os.path.exists(file):
            print("Config file was not found under %s. Default file has been created" % CONFIG_FILE)
            self._settings = yaml.load(DEFAULT_CONFIG, yaml.RoundTripLoader)
            self.save(file)
            sys.exit()
        with open(file, 'r') as f:
            self._settings = yaml.load(f, yaml.RoundTripLoader)