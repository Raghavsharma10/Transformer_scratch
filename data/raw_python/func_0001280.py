def read_file(self, file_path):
        ''' Read a configuration file and return configuration data '''
        getLogger().info("Loading app config from {} file: {}".format(self.__mode, file_path))
        if self.__mode == AppConfig.JSON:
            return json.loads(FileHelper.read(file_path), object_pairs_hook=OrderedDict)
        elif self.__mode == AppConfig.INI:
            config = configparser.ConfigParser(allow_no_value=True)
            config.read(file_path)
            return config