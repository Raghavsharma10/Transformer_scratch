def add(self, config_files, galaxy_root=None):
        """ Public method to add (register) config file(s).
        """
        for config_file in config_files:
            config_file = abspath(expanduser(config_file))
            if self.is_registered(config_file):
                log.warning('%s is already registered', config_file)
                continue
            defaults = None
            if galaxy_root is not None:
                defaults={ 'galaxy_root' : galaxy_root }
            conf = ConfigManager.get_ini_config(config_file, defaults=defaults)
            if conf is None:
                raise Exception('Cannot add %s: File is unknown type' % config_file)
            if conf['instance_name'] is None:
                conf['instance_name'] = conf['config_type'] + '-' + hashlib.md5(os.urandom(32)).hexdigest()[:12]
            if conf['attribs']['virtualenv'] is None:
                conf['attribs']['virtualenv'] = abspath(join(expanduser(self.state_dir), 'virtualenv-' + conf['instance_name']))
            # create the virtualenv if necessary
            self.create_virtualenv(conf['attribs']['virtualenv'])
            conf_data = { 'config_type' : conf['config_type'],
                          'instance_name' : conf['instance_name'],
                          'attribs' : conf['attribs'],
                          'services' : [] } # services will be populated by the update method
            self._register_config_file(config_file, conf_data)
            log.info('Added %s config: %s', conf['config_type'], config_file)