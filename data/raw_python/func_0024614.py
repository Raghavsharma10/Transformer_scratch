def read_config(self, path):
        """Read configuration file."""
        PYVLXLOG.info('Reading config file: %s', path)
        try:
            with open(path, 'r') as filehandle:
                doc = yaml.safe_load(filehandle)
                self.test_configuration(doc, path)
                self.host = doc['config']['host']
                self.password = doc['config']['password']
                if 'port' in doc['config']:
                    self.port = doc['config']['port']
        except FileNotFoundError as ex:
            raise PyVLXException('file does not exist: {0}'.format(ex))