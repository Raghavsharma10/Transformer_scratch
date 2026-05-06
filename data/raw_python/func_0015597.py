def pretty_print_json(self, json_string):
        """
        Return formatted JSON string _json_string_.\n
        Using method json.dumps with settings: _indent=2, ensure_ascii=False_.

        *Args:*\n
        _json_string_ - JSON string.

        *Returns:*\n
        Formatted JSON string.

        *Example:*\n
        | *Settings* | *Value* |
        | Library    | JsonValidator |
        | Library    | OperatingSystem |
        | *Test Cases* | *Action* | *Argument* | *Argument* |
        | Check element | ${pretty_json}=   | Pretty print json |   {a:1,foo:[{b:2,c:3},{d:"baz",e:4}]} |
        | | Log  |  ${pretty_json}  |
        =>\n
        | {
        |    "a": 1,
        |    "foo": [
        |      {
        |        "c": 3,
        |        "b": 2
        |      },
        |      {
        |        "e": 4,
        |        "d": "baz"
        |      }
        |    ]
        | }
        """
        return json.dumps(self.string_to_json(json_string), indent=2, ensure_ascii=False)