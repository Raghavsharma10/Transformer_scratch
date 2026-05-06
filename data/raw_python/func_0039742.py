def json_schema_validation_format(value, schema_validation_type):
    """
    adds iso8601 to the datetimevalidator
    raises SchemaError if validation fails
    """
    DEFAULT_FORMAT_VALIDATORS['date-time'] = validate_format_iso8601
    DEFAULT_FORMAT_VALIDATORS['text'] = validate_format_text
    validictory.validate(value, schema_validation_type, format_validators=DEFAULT_FORMAT_VALIDATORS)