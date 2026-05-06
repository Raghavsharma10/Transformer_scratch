def select_elements(self, json_string, expr):
        """
        Return list of elements from _json_string_, matching [ http://jsonselect.org/ | JSONSelect] expression.

        *DEPRECATED* JSON Select query language is outdated and not supported any more.
        Use other keywords of this library to query JSON.

        *Args:*\n
        _json_string_ - JSON string;\n
        _expr_ - JSONSelect expression;

        *Returns:*\n
        List of found elements or ``None`` if no elements were found

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Select json elements | ${json_example}=   | OperatingSystem.Get File |   ${CURDIR}${/}json_example.json |
        |                      |  ${json_elements}= | Select elements  |  ${json_example}  |  .author:contains("Evelyn Waugh")~.price |
        =>\n
        | 12.99
        """
        load_input_json = self.string_to_json(json_string)
        # parsing jsonselect
        match = jsonselect.match(sel=expr, obj=load_input_json)
        ret = list(match)
        return ret if ret else None