def _generate_url(self, regex, arguments):
        """
        Uses the regex (of the type defined in Django's url patterns) and the arguments to return a relative URL
        For example, if the regex is '^/api/shreddr/job/(?P<id>[\d]+)$' and arguments is ['23']
        then return would be '/api/shreddr/job/23'
        """
        regex_tokens = _split_regex(regex)
        result = ''
        for i in range(len(arguments)):
            result = result + str(regex_tokens[i]) + str(arguments[i])
        if len(regex_tokens) > len(arguments):
            result += regex_tokens[-1]
        return result