def check_immediate(self, arg):
        """
        Is the parameter an immediate in the form of '#<d>',

        Raises an exception if
        1. The parameter is not in the form of '#<d>'
        :param arg: The parameter to check
        :return: The value of the immediate
        """
        self.check_parameter(arg)
        match = re.search(self.IMMEDIATE_REGEX, arg)
        if match is None:
            raise iarm.exceptions.RuleError("Parameter {} is not an immediate".format(arg))
        return self.convert_to_integer(match.groups()[0])