def get_settings(self, section=None, defaults=None):
        """
        Gets a named section from the configuration source.

        :param section: a :class:`str` representing the section you want to
            retrieve from the configuration source. If ``None`` this will
            fallback to the :attr:`plaster.PlasterURL.fragment`.
        :param defaults: a :class:`dict` that will get passed to
            :class:`configparser.ConfigParser` and will populate the
            ``DEFAULT`` section.
        :return: A :class:`plaster_pastedeploy.ConfigDict` of key/value pairs.

        """
        # This is a partial reimplementation of
        # ``paste.deploy.loadwsgi.ConfigLoader:get_context`` which supports
        # "set" and "get" options and filters out any other globals
        section = self._maybe_get_default_name(section)
        if self.filepath is None:
            return {}
        parser = self._get_parser(defaults)
        defaults = parser.defaults()

        try:
            raw_items = parser.items(section)
        except NoSectionError:
            return {}

        local_conf = OrderedDict()
        get_from_globals = {}
        for option, value in raw_items:
            if option.startswith("set "):
                name = option[4:].strip()
                defaults[name] = value
            elif option.startswith("get "):
                name = option[4:].strip()
                get_from_globals[name] = value
                # insert a value into local_conf to preserve the order
                local_conf[name] = None
            else:
                # annoyingly pastedeploy filters out all defaults unless
                # "get foo" is used to pull it in
                if option in defaults:
                    continue
                local_conf[option] = value
        for option, global_option in get_from_globals.items():
            local_conf[option] = defaults[global_option]
        return ConfigDict(local_conf, defaults, self)