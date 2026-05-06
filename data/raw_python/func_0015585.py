def _validate_json(self, checked_json, schema):
        """ Validate JSON according to JSONSchema

        *Args*:\n
        _checked_json_: validated JSON.
        _schema_: schema that used for validation.
        """
        try:
            jsonschema.validate(checked_json, schema)
        except jsonschema.ValidationError as e:
            print("""Failed validating '{0}'
in schema {1}:
{2}
On instance {3}:
{4}""".format(e.validator,
              list(e.relative_schema_path)[:-1], pprint.pformat(e.schema),
              "[%s]" % "][".join(repr(index) for index in e.absolute_path),
              pprint.pformat(e.instance).encode('utf-8')))
            raise JsonValidatorError("Failed validating json by schema")
        except jsonschema.SchemaError as e:
            raise JsonValidatorError('Json-schema error')