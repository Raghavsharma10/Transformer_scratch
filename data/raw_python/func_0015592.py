def select_objects(self, json_string, expr):
        """
        Return list of elements from _json_string_, matching [ http://objectpath.org// | ObjectPath] expression.

        *Args:*\n
        _json_string_ - JSON string;\n
        _expr_ - ObjectPath expression;

        *Returns:*\n
        List of found elements. If no elements were found, empty list will be returned

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Select json objects  | ${json_example}=   | OperatingSystem.Get File |   ${CURDIR}${/}json_example.json |
        |                      |  ${json_objectss}= | Select objects  |  ${json_example}  |  $..book[@.author.name is "Evelyn Waugh"].price |
        =>\n
        | [12.99]
        """
        load_input_json = self.string_to_json(json_string)
        # parsing objectpath
        tree = objectpath.Tree(load_input_json)
        values = tree.execute(expr)
        return list(values)