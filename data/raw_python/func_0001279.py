def config(self):
        ''' Read config automatically if required '''
        if self.__config is None:
            config_path = self.locate_config()
            if config_path:
                self.__config = self.read_file(config_path)
                self.__config_path = config_path
        return self.__config