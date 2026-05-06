def modeltag_nonref_schemas(spec):
    """
    This function will go through the OpenAPI 'paths' and look for any
    command parameters that have non "$ref" schemas defined.  If the
    parameter does have a $ref schema, then the bravado library will
    do this x-model tagging.  But it does not do it for schemas that
    are defined inside the path/command structure

    Parameters
    ----------
    spec : dict
        The OpenApi spec dictionary
    """

    for path_name, path_data in six.iteritems(spec['paths']):
        for path_cmd, cmd_data in six.iteritems(path_data):

            # check the parameters, looking for "schemas" that
            # do not have $ref or already tagged.  when found
            # use either the operationId or the (command, api-path)
            # to formulate a humanized tag name.

            for param in cmd_data.get('parameters'):
                schema = param.get('schema')
                if schema and ('$ref' not in schema) and (MODEL_MARKER not in schema):
                    model_name = (camelize(cmd_data.get('operationId')) or
                                  "%s%s" % (path_cmd.upper(),
                                            humanize_api_path(path_name)))

                    schema[MODEL_MARKER] = "%s%s" % (model_name, MODEL_NAME_SUFFIX)