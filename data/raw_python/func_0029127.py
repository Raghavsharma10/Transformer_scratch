def check_arguments(self, **kwargs):
        """
        Determine if the parameters meet the specifications
        kwargs contains lists grouped by their parameter
        rules are defined by methods starting with 'rule_'
        :param kwargs:
        :return:
        """
        for key in kwargs:
            if key in self._rules:
                for val in kwargs[key]:
                    self._rules[key](val)
            else:
                raise LookupError("Rule for {} does not exist. Make sure the rule starts with 'rule_'".format(key))