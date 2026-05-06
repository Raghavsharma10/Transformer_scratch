def add_argument(self, *args, **kwargs):
        """ Add an argument incorporating the default value into the help string

        :param args:
        :param kwargs:
        :return:
        """
        defhelp = kwargs.pop("help", None)
        defaults = kwargs.pop("default", None)
        default = defaults if self.use_defaults else None
        if not defhelp or default is None or kwargs.get('action') == 'help':
            return super().add_argument(*args, help=defhelp, default=default, **kwargs)
        else:
            return super().add_argument(*args, help=defhelp + " (default: {})".format(default),
                                        default=default, **kwargs)