def get_default_name(self):
        '''
        Return the default generated name to store value on the parser for this option.

        eg. An option *['-s', '--use-ssl']* will generate the *use_ssl* name

        Returns:
            str: the default name of the option
        '''
        long_names = [name for name in self.name if name.startswith("--")]
        short_names = [name for name in self.name if not name.startswith("--")]

        if long_names:
            return to_snake_case(long_names[0].lstrip("-"))

        return to_snake_case(short_names[0].lstrip("-"))