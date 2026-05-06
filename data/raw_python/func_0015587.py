def validate_jsonschema(self, json_string, input_schema):
        """
        Validate JSON according to schema.

        *Args:*\n
        _json_string_ - JSON string;\n
        _input_schema_ - schema in string format;

        *Raises:*\n
        JsonValidatorError

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Simple | ${schema}=   | OperatingSystem.Get File |   ${CURDIR}${/}schema_valid.json |
        |  | Validate jsonschema  |  {"foo":bar}  |  ${schema} |
        """
        load_input_json = self.string_to_json(json_string)

        try:
            load_schema = json.loads(input_schema)
        except ValueError as e:
            raise JsonValidatorError('Error in schema: {}'.format(e))

        self._validate_json(load_input_json, load_schema)