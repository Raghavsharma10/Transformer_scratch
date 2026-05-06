def build_defaults(self):
        """Build a dictionary of default values from the `Scheme`.

        Returns:
            dict: The default configurations as set by the `Scheme`.

        Raises:
            errors.InvalidSchemeError: The `Scheme` does not contain
                valid options.
        """
        defaults = {}
        for arg in self.args:
            if not isinstance(arg, _BaseOpt):
                raise errors.InvalidSchemeError('Unable to build default for non-Option type')

            # if there is a default set, add it to the defaults dict
            if not isinstance(arg.default, NoDefault):
                defaults[arg.name] = arg.default

            # if we have a dict option, build the defaults for its scheme.
            # if any defaults exist, use them.
            if isinstance(arg, DictOption):
                if arg.scheme:
                    b = arg.scheme.build_defaults()
                    if b:
                        defaults[arg.name] = b
        return defaults