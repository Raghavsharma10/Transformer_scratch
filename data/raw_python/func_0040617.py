def loads(cls, json_text, schema=None):
        """
        :param str json_text: json text to be parse
        :param voluptuous.Schema schema: JSON schema.
        :return: Dictionary storing the parse results of JSON
        :rtype: dictionary
        :raises ImportError:
        :raises RuntimeError:
        :raises ValueError:
        """

        try:
            json_text = json_text.decode("ascii")
        except AttributeError:
            pass

        try:
            dict_json = json.loads(json_text)
        except ValueError:
            _, e, _ = sys.exc_info()  # for python 2.5 compatibility
            raise ValueError(os.linesep.join([
                str(e),
                "decode error: check JSON format with http://jsonlint.com/",
            ]))

        cls.__validate_json(schema, dict_json)

        return dict_json