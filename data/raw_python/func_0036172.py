def _create_default_config_file(self):
        """
        If config file does not exists create and set default values.
        """
        logger.info('Initialize Maya launcher, creating config file...\n')
        self.add_section(self.DEFAULTS)
        self.add_section(self.PATTERNS)
        self.add_section(self.ENVIRONMENTS)
        self.add_section(self.EXECUTABLES)
        self.set(self.DEFAULTS, 'executable', None)
        self.set(self.DEFAULTS, 'environment', None)
        self.set(self.PATTERNS, 'exclude', ', '.join(self.EXLUDE_PATTERNS))
        self.set(self.PATTERNS, 'icon_ext', ', '.join(self.ICON_EXTENSIONS))

        self.config_file.parent.mkdir(exist_ok=True)
        self.config_file.touch()
        with self.config_file.open('wb') as f:
            self.write(f)

        # If this function is run inform the user that a new file has been
        # created.
        sys.exit('Maya launcher has successfully created config file at:\n'
                 ' "{}"'.format(str(self.config_file)))