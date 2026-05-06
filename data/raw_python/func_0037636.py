def read(self, config_dir=None, clear=False, config_file=None):
        """
        The munge Config's read function only allows to read from
        a config directory, but we also want to be able to read
        straight from a config file as well
        """

        if config_file:
            data_file = os.path.basename(config_file)
            data_path = os.path.dirname(config_file)

            if clear:
                self.clear()

            config = munge.load_datafile(data_file, data_path, default=None)

            if not config:
                raise IOError("Config file not found: %s" % config_file)

            munge.util.recursive_update(self.data, config)
            self._meta_config_dir = data_path
            return
        else:
            return super(Config, self).read(config_dir=config_dir, clear=clear)