def config(self):
        """load the passwords from the config file
        """
        if not hasattr(self, '_config'):
            raw_config = configparser.RawConfigParser()
            f = self._open()
            if f:
                raw_config.readfp(f)
                f.close()
            self._config = raw_config
        return self._config