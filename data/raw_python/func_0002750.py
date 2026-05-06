def _red_listing_validation(key, listing):
    """
    Raises an RedValidationError, if the given listing does not comply with cwl_job_listing_schema.
    If listing is None or an empty list, no exception is thrown.

    :param key: The input key to build an error message if needed.
    :param listing: The listing to validate
    :raise RedValidationError: If the given listing does not comply with cwl_job_listing_schema
    """

    if listing:
        try:
            jsonschema.validate(listing, cwl_job_listing_schema)
        except ValidationError as e:
            raise RedValidationError('REDFILE listing of input "{}" does not comply with jsonschema: {}'
                                     .format(key, e.context))