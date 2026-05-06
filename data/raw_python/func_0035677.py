def load(self, path=None):
        '''
        Load configuration (from configuration files).

        Parameters
        ----------
        path : ~pathlib.Path or None
            Path to configuration file, which must exist; or path to directory
            containing a configuration file; or None.

        Returns
        -------
        ~typing.Dict[str, ~typing.Dict[str, str]]
            The configuration as a dict of sections mapping section name to
            options. Each options dict maps from option name to option value. The
            ``default`` section is not included. However, all options from the
            ``default`` section are included in each returned section.

        Raises
        ------
        ValueError
            If ``path`` is a missing file; or if it is a directory which does not
            contain the configuration file.

        Examples
        --------
        >>> loader.load()
        {
            'section1': {
                'option1': 'value',
                'option2': 'value2',
            }
        }
        '''
        # Add path
        paths = self._paths.copy()
        if path:
            if path.is_dir():
                path /= '{}.conf'.format(self._configuration_name)
            paths.append(path)

        # Prepend file sys root to abs paths
        paths = [(path_._root / str(x)[1:] if x.is_absolute() else x) for x in paths]
        if path:
            path = paths[-1]

            # Passed path must exist
            if not path.exists():
                raise ValueError('Expected configuration file at {}'.format(path))

        # Configure parser
        config_parser = ConfigParser(
            inline_comment_prefixes=('#', ';'), 
            empty_lines_in_values=False, 
            default_section='default', 
            interpolation=ExtendedInterpolation()
        )

        def option_transform(name):
            return name.replace('-', '_').replace(' ', '_').lower()

        config_parser.optionxform = option_transform

        # Parse defaults and configs
        with suppress(FileNotFoundError):
            defaults_contents = resource_string(self._package_name, 'data/{}.defaults.conf'.format(self._configuration_name))
            config_parser.read_string(defaults_contents.decode('UTF-8'))
        config_parser.read([str(x) for x in paths])  # reads in given order

        config = {k : dict(v) for k,v in config_parser.items()}
        del config['default']
        return config