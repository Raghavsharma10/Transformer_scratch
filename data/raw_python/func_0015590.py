def get_elements(self, json_string, expr):
        """
        Get list of elements from _json_string_, matching [http://goessner.net/articles/JsonPath/|JSONPath] expression.

        *Args:*\n
        _json_string_ - JSON string;\n
        _expr_ - JSONPath expression;

        *Returns:*\n
        List of found elements or ``None`` if no elements were found

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Get json elements | ${json_example}=   | OperatingSystem.Get File |   ${CURDIR}${/}json_example.json |
        |                   |  ${json_elements}= | Get elements  |  ${json_example}  |  $.store.book[*].author |
        =>\n
        | [u'Nigel Rees', u'Evelyn Waugh', u'Herman Melville', u'J. R. R. Tolkien']
        """
        load_input_json = self.string_to_json(json_string)
        # parsing jsonpath
        jsonpath_expr = parse(expr)
        # list of returned elements
        value_list = []
        for match in jsonpath_expr.find(load_input_json):
            value_list.append(match.value)
        if not value_list:
            return None
        else:
            return value_list