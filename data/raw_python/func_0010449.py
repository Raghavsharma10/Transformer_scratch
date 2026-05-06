def get_filter_options(cls):
        """
        List all filter options defined on class (and superclasses)
        """
        attr = '_filter_options_%s' % id(cls)

        options = getattr(cls, attr, {})
        if options:
            return options

        for key in dir(cls):
            val = getattr(cls, key)
            if isinstance(val, FilterOpt):
                options[key] = val

        setattr(cls, attr, options)
        return options