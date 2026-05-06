def get_parameters(self, regex_exp, parameters):
        """
        Given a regex expression and the string with the paramers,
        either return a regex match object or raise an exception if the regex
        did not find a match
        :param regex_exp:
        :param parameters:
        :return:
        """
        # TODO find a better way to do the equate replacement
        for rep in self.equates:
            parameters = parameters.replace(rep, str(self.equates[rep]))
        match = re.match(regex_exp, parameters)
        if not match:
            raise iarm.exceptions.ParsingError("Parameters are None, did you miss a comma?")

        return match.groups()