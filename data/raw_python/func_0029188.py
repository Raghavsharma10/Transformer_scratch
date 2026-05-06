def get_three_parameters(self, regex_exp, parameters):
        """
        Get three parameters from a given regex expression

        Raise an exception if more than three were found
        :param regex_exp:
        :param parameters:
        :return:
        """
        Rx, Ry, Rz, other = self.get_parameters(regex_exp, parameters)
        if other is not None and other.strip():
            raise iarm.exceptions.ParsingError("Extra arguments found: {}".format(other))
        return Rx.upper(), Ry.upper(), Rz.upper()