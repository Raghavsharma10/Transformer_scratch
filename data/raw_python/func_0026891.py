def parse_args(self):
        """
        Called from ``gnotty.server.run`` and parses any CLI args
        provided. Also handles loading settings from the Python
        module specified with the ``--conf-file`` arg. CLI args
        take precedence over any settings defined in the Python
        module defined by ``--conf-file``.
        """
        options, _ = parser.parse_args()
        file_settings = {}
        if options.CONF_FILE:
            execfile(options.CONF_FILE, {}, file_settings)
        for option in self.option_list:
            if option.dest:
                file_value = file_settings.get("GNOTTY_%s" % option.dest, None)
                # optparse doesn't seem to provide a way to determine if
                # an option's value was provided as a CLI arg, or if the
                # default is being used, so we manually check sys.argv,
                # since provided CLI args should take precedence over
                # any settings defined in a conf module.
                flags = option._short_opts + option._long_opts
                in_argv = set(flags) & set(sys.argv)
                options_value = getattr(options, option.dest)
                if file_value and not in_argv:
                    self[option.dest] = file_value
                elif in_argv:
                    self[option.dest] = options_value
                else:
                    self[option.dest] = self.get(option.dest, options_value)
        self.set_max_message_length()
        self["STATIC_URL"] = "/static/"
        self["LOG_LEVEL"] = getattr(logging, self["LOG_LEVEL"])