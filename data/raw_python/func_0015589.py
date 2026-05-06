def json_to_string(self, source):
        """
        Serialize JSON structure into string.

        *Args:*\n
        _source_ - JSON structure

        *Returns:*\n
        JSON string

        *Raises:*\n
        JsonValidatorError

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Json to string  | ${json_string}=   | OperatingSystem.Get File |   ${CURDIR}${/}json_example.json |
        |                 | ${json}= | String to json |   ${json_string} |
        |                 | ${string}=  |  Json to string  |  ${json} |
        |                 | ${pretty_string}=  |  Pretty print json  |  ${string} |
        |                 | Log to console  |  ${pretty_string} |
        """
        try:
            load_input_json = json.dumps(source)
        except ValueError as e:
            raise JsonValidatorError("Could serialize '%s' to JSON: %s" % (source, e))
        return load_input_json