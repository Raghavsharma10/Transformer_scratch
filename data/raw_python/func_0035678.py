def cli_help_message(self, description):
        '''
        Get a user friendly help message that can be dropped in a
        `click.Command`\ 's epilog.

        Parameters
        ----------
        description : str
            Description of the configuration file to include in the message.

        Returns
        -------
        str
            A help message that uses :py:mod:`click`\ 's help formatting
            constructs (e.g. ``\b``).
        '''
        config_files_listing = '\n'.join('    {}. {!s}'.format(i, path) for i, path in enumerate(self._paths, 1))
        text = dedent('''\
        {config_file}:
        
            {description}
            
            {config_file} files are read from the following locations:
            
            \b
            {config_files_listing}
            
            Any configuration file can override options set by previous configuration files. Some 
            configuration file locations can be changed using the XDG standard (http://standards.freedesktop.org/basedir-spec/basedir-spec-0.6.html).
        ''').format(
            config_file='{}.conf'.format(self._configuration_name),
            description=description,
            config_files_listing=config_files_listing
        )
        return text