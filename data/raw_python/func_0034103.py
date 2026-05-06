def getKwargs(self, args, values={}, get=Get()):
        """Gets necessary data from user input.

        :args: Dictionary of arguments supplied in command line.
        :values: Default values dictionary, supplied for editing.
        :get: Object used to get values from user input.
        :returns: A dictionary containing data gathered from user input.

        """
        kwargs = dict()
        for field in ['name', 'priority', 'comment', 'parent']:
            fvalue = args.get(field) or get.get(field, values.get(field))
            if fvalue is not None:
                kwargs[field] = fvalue
        return kwargs