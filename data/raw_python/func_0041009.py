def parse_options(cls, options):
        """Required by flake8
        parse the options, called after add_options

        Args:
            options (dict): options to be parsed
        """
        d = {}
        for filename_check, dictionary in cls.filename_checks.items():
            # retrieve the marks from the passed options
            filename_data = getattr(options, filename_check)
            if len(filename_data) != 0:
                parsed_params = {}
                for single_line in filename_data:
                    a = [s.strip() for s in single_line.split('=')]
                    # whitelist the acceptable params
                    if a[0] in ['filter_regex', 'filename_regex']:
                        parsed_params[a[0]] = a[1]
                d[filename_check] = parsed_params
        cls.filename_checks.update(d)
        # delete any empty rules
        cls.filename_checks = {x: y for x, y in cls.filename_checks.items() if len(y) > 0}