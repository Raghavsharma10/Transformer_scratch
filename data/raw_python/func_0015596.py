def update_json(self, json_string, expr, value, index=0):
        """
        Replace the value in the JSON string.

        *Args:*\n
        _json_string_ - JSON string;\n
        _expr_ - JSONPath expression for determining the value to be replaced;\n
        _value_ - the value to be replaced with;\n
        _index_ - index for selecting item within a match list, default value is 0;\n

        *Returns:*\n
        Changed JSON in dictionary format.

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Update element | ${json_example}=   | OperatingSystem.Get File |   ${CURDIR}${/}json_example.json |
        | | ${json_update}= | Update_json  |  ${json_example}  |  $..color  |  changed |
        """
        load_input_json = self.string_to_json(json_string)
        matches = self._json_path_search(load_input_json, expr)

        datum_object = matches[int(index)]

        if not isinstance(datum_object, DatumInContext):
            raise JsonValidatorError("Nothing found by the given json-path")

        path = datum_object.path

        # Edit the directory using the received data
        # If the user specified a list
        if isinstance(path, Index):
            datum_object.context.value[datum_object.path.index] = value
        # If the user specified a value of type (string, bool, integer or complex)
        elif isinstance(path, Fields):
            datum_object.context.value[datum_object.path.fields[0]] = value

        return load_input_json