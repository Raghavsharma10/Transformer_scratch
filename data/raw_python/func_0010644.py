def consolidate(self, args):
        """ Consolidate the provided arguments.

        If the provided arguments have matching options, this performs a type conversion.
        For any option that has a default value and is not present in the provided
        arguments, the default value is added.

        Args:
            args (dict): A dictionary of the provided arguments.

        Returns:
            dict: A dictionary with the type converted and with default options enriched
                  arguments.
        """
        result = dict(args)

        for opt in self:
            if opt.name in result:
                result[opt.name] = opt.convert(result[opt.name])
            else:
                if opt.default is not None:
                    result[opt.name] = opt.convert(opt.default)

        return result