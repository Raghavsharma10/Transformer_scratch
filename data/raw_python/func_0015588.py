def string_to_json(self, source):
        """
        Deserialize string into JSON structure.

        *Args:*\n
        _source_ - JSON string

        *Returns:*\n
        JSON structure

        *Raises:*\n
        JsonValidatorError

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | String to json  | ${json_string}=   | OperatingSystem.Get File |   ${CURDIR}${/}json_example.json |
        |                 |  ${json}= | String to json  |  ${json_string} |
        |                 |  Log | ${json["store"]["book"][0]["price"]} |
        =>\n
        8.95
        """
        try:
            load_input_json = json.loads(source)
        except ValueError as e:
            raise JsonValidatorError("Could not parse '%s' as JSON: %s" % (source, e))
        return load_input_json