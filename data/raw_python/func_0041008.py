def add_options(cls, parser):
        """Required by flake8
        add the possible options, called first

        Args:
            parser (OptionsManager):
        """
        kwargs = {'action': 'store', 'default': '', 'parse_from_config': True,
                  'comma_separated_list': True}
        for num in range(cls.min_check, cls.max_check):
            parser.add_option(None, "--filename_check{}".format(num), **kwargs)