def _load_yaml_config(self):
        """Loads the configuration file from a .yaml or .yml file

        :type: dict

        """
        try:
            config = self._read_config()
        except OSError as error:
            raise ValueError('Could not read configuration file: %s' % error)
        try:
            return yaml.safe_load(config)
        except yaml.YAMLError as error:
            message = '\n'.join(['    > %s' % line
                                 for line in str(error).split('\n')])
            sys.stderr.write('\n\n  Error in the configuration file:\n\n'
                             '{}\n\n'.format(message))
            sys.stderr.write('  Configuration should be a valid YAML file.\n')
            sys.stderr.write('  YAML format validation available at '
                             'http://yamllint.com\n')
            raise ValueError(error)