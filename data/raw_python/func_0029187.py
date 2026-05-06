def get_two_parameters(self, regex_exp, parameters):
        """
        Get two parameters from a given regex expression

        Raise an exception if more than two were found
        :param regex_exp:
        :param parameters:
        :return:
        """
        Rx, Ry, other = self.get_parameters(regex_exp, parameters)
        if other is not None and other.strip():
            raise iarm.exceptions.ParsingError("Extra arguments found: {}".format(other))
        if Rx and Ry:
            return Rx.upper(), Ry.upper()
        elif not Rx:
            raise iarm.exceptions.ParsingError("Missing first positional argument")
        else:
            raise iarm.exceptions.ParsingError("Missing second positional argument")