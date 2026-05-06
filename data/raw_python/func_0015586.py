def validate_jsonschema_from_file(self, json_string, path_to_schema):
        """
        Validate JSON according to schema, loaded from a file.

        *Args:*\n
        _json_string_ - JSON string;\n
        _path_to_schema_ - path to file with JSON schema;

        *Raises:*\n
        JsonValidatorError

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Simple | Validate jsonschema from file  |  {"foo":bar}  |  ${CURDIR}${/}schema.json |
        """
        schema = open(path_to_schema).read()
        load_input_json = self.string_to_json(json_string)

        try:
            load_schema = json.loads(schema)
        except ValueError as e:
            raise JsonValidatorError('Error in schema: {}'.format(e))

        self._validate_json(load_input_json, load_schema)