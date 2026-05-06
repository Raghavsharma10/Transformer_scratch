def save_config(self, config_file_name):
        """ Save configuration file to xpc file.

        :param config_file_name: full path to the configuration file.
        """

        with open(config_file_name, 'w+') as f:
            f.write('P_RESET\n')
            for line in self.send_command_return_multilines('p_fullconfig', '?'):
                f.write(line.split(' ', 1)[1].lstrip())