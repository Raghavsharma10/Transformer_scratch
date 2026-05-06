def load_config(self, config_file_name):
        """ Load configuration file from prt or str.

        Configuration file type is extracted from the file suffix - prt or str.

        :param config_file_name: full path to the configuration file.
            IxTclServer must have access to the file location. either:
                The config file is on shared folder.
                IxTclServer run on the client machine.
        """
        config_file_name = config_file_name.replace('\\', '/')
        ext = path.splitext(config_file_name)[-1].lower()
        if ext == '.prt':
            self.api.call_rc('port import "{}" {}'.format(config_file_name, self.uri))
        elif ext == '.str':
            self.reset()
            self.api.call_rc('stream import "{}" {}'.format(config_file_name, self.uri))
        else:
            raise ValueError('Configuration file type {} not supported.'.format(ext))
        self.write()
        self.discover()