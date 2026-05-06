def element_should_exist(self, json_string, expr):
        """
        Check the existence of one or more elements, matching [ http://jsonselect.org/ | JSONSelect] expression.

        *DEPRECATED* JSON Select query language is outdated and not supported any more.
        Use other keywords of this library to query JSON.

        *Args:*\n
        _json_string_ - JSON string;\n
        _expr_ - JSONSelect expression;\n

        *Raises:*\n
        JsonValidatorError

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Check element | ${json_example}=   | OperatingSystem.Get File |   ${CURDIR}${/}json_example.json |
        | | Element should exist  |  ${json_example}  |  .author:contains("Evelyn Waugh") |
        | | Element should exist  |  ${json_example}  |  .store .book  .price:expr(x=8.95) |
        """
        value = self.select_elements(json_string, expr)
        if value is None:
            raise JsonValidatorError('Elements %s does not exist' % expr)