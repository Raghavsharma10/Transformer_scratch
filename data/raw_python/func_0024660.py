def read_config(self, path):
        """Read configuration file."""
        self.pyvlx.logger.info('Reading config file: ', path)
        try:
            with open(path, 'r') as filehandle:
                doc = yaml.load(filehandle)
                if 'config' not in doc:
                    raise PyVLXException('no element config found in: {0}'.format(path))
                if 'host' not in doc['config']:
                    raise PyVLXException('no element host found in: {0}'.format(path))
                if 'password' not in doc['config']:
                    raise PyVLXException('no element password found in: {0}'.format(path))
                self.host = doc['config']['host']
                self.password = doc['config']['password']
        except FileNotFoundError as ex:
            raise PyVLXException('file does not exist: {0}'.format(ex))